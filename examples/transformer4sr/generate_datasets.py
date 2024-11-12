# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://github.com/omron-sinicx/transformer4sr
"""


import concurrent.futures
import os
import warnings
from functools import partial

import hydra
import numpy as np
import sympy
from omegaconf import DictConfig
from tqdm import tqdm
from utils import MY_VOCAB
from utils import count_var_num
from utils import expr_tree_depth
from utils import from_seq_to_sympy
from utils import from_sympy_to_seq
from utils import gen_expr
from utils import gen_samples
from utils import reassign_variables

import ppsci  # noqa

warnings.filterwarnings("ignore")


def fliter_nodes(expr, num_nodes):
    if num_nodes[0] <= len(expr) <= num_nodes[1]:
        return expr
    else:
        return None


def fliter_nested(expr, num_nested_max):
    try:
        expr_sympy = from_seq_to_sympy(expr)
        expr_sympy = sympy.factor(expr_sympy)
        expr_sympy = sympy.simplify(expr_sympy)
        assert "zoo" not in str(expr_sympy)
        assert expr_tree_depth(expr_sympy) <= num_nested_max
        expr_sympy = reassign_variables(expr_sympy)
        expr_sympy = sympy.factor(expr_sympy)
        expr_sympy = sympy.simplify(expr_sympy)
        return expr_sympy
    except Exception:
        return None


def fliter_consts_vars_len(expr, num_consts, num_vars, seq_length_max):
    try:
        cnt_const = expr.count("C")
        assert "abort" not in expr
        assert num_consts[0] <= cnt_const <= num_consts[1]
        assert f"x{num_vars[0]}" in expr
        assert f"x{num_vars[1] + 1}" not in expr
        assert len(expr) <= seq_length_max
        return expr
    except Exception:
        return None


def save_dataset(dataset, ground_truth, value_path, gt_path):
    np.save(value_path, dataset)
    with open(
        gt_path,
        "w",
    ) as f:
        for token in ground_truth:
            f.write(f"{token}\n")


def generate_data(cfg: DictConfig):
    # init trees
    exprs_init = []
    num_init_trials = cfg.DATA_GENERATE.num_init_trials
    for i in tqdm(range(num_init_trials), desc="Initial expression trees"):
        exprs_init.append(gen_expr(MY_VOCAB))

    # fliter nodes
    num_nodes = cfg.DATA_GENERATE.num_nodes
    exprs_filter_nodes = []
    for expr in tqdm(exprs_init, desc="Check nodes number"):
        expr = fliter_nodes(expr, num_nodes)
        if expr is not None:
            exprs_filter_nodes.append(expr)

    # fliter nested
    num_nested_max = cfg.DATA_GENERATE.num_nested_max
    partial_fliter_nested = partial(fliter_nested, num_nested_max=num_nested_max)
    exprs_fliter_nested = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_expr = {
            executor.submit(partial_fliter_nested, expr): expr
            for expr in exprs_filter_nodes
        }
        progress = tqdm(
            concurrent.futures.as_completed(future_to_expr),
            total=len(exprs_filter_nodes),
            desc=f"Check invalid abd very nested (>{num_nested_max}) expressions",
        )
        for future in progress:
            expr = future_to_expr[future]
            try:
                expr_sympy = future.result()
                if expr_sympy is not None:
                    exprs_fliter_nested.append(expr_sympy)
            except Exception:
                continue

    # fliter consts/vars/seq_length
    num_consts = cfg.DATA_GENERATE.num_consts
    num_vars = cfg.DATA_GENERATE.num_vars
    seq_length_max = cfg.DATA_GENERATE.seq_length_max
    exprs_cvl = []
    for i in tqdm(range(len(exprs_fliter_nested)), desc="Check consts and vars."):
        expr_seq = from_sympy_to_seq(exprs_fliter_nested[i])
        expr_seq = fliter_consts_vars_len(
            expr_seq, num_consts, num_vars, seq_length_max
        )
        if expr_seq is not None:
            exprs_cvl.append(expr_seq)

    unique_expr_tuples = {tuple(expr) for expr in exprs_cvl}
    expr_uniq_seq = [list(expr) for expr in unique_expr_tuples]

    # generate datasets
    num_sampling_per_eq = cfg.DATA_GENERATE.num_sampling_per_eq
    sampling_times = cfg.DATA_GENERATE.sampling_times
    order_of_mag_limit = cfg.DATA_GENERATE.order_of_mag_limit
    var_type = cfg.DATA_GENERATE.var_type
    num_zfill = cfg.DATA_GENERATE.num_zfill
    out_dir = cfg.DATA_GENERATE.data_path
    gt_dir = os.path.join(out_dir, "ground_truth")
    value_dir = os.path.join(out_dir, "values")
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(value_dir):
        os.makedirs(value_dir)

    count_datasets = 0
    for uniq_seq in tqdm(expr_uniq_seq, desc="Generate datasets"):
        try:
            for _ in tqdm(
                range(num_sampling_per_eq), desc="Generate samples", leave=False
            ):
                seq_deformed = []
                ground_truth = []
                for token in uniq_seq:
                    if token == "C":
                        const_val = np.round(
                            np.random.uniform(low=-100.0, high=100.0), decimals=2
                        )
                        seq_deformed.append(str(const_val))
                        ground_truth.append(f"C={str(const_val)}")
                    else:
                        seq_deformed.append(token)
                        ground_truth.append(token)

                cur_sympy_expr = from_seq_to_sympy(seq_deformed)
                np_y, np_x = gen_samples(cur_sympy_expr, num_samples=1000)
                assert np.nanmax(np.abs(np_y)) <= order_of_mag_limit
                mask = np.logical_not(np.isnan(np_y))
                num_temp_obs = np.sum(mask)
                assert num_temp_obs >= sampling_times

                idx = np.random.choice(num_temp_obs, size=sampling_times, replace=False)
                num_var = count_var_num(sampling_times)
                x_values = np_x[mask][idx, :num_var]
                y_values = np_y[mask][idx]
                if var_type == "both":
                    dataset = np.zeros((sampling_times, 14))
                else:
                    dataset = np.zeros((sampling_times, 7))

                if var_type == "normal":
                    dataset[:, 0] = y_values
                    dataset[:, 1 : num_var + 1] = x_values
                elif var_type == "log":
                    dataset[:, 0] = np.log(np.abs(y_values) + 1e-10)
                    dataset[:, 1 : num_var + 1] = np.log(np.abs(x_values) + 1e-10)
                elif var_type == "both":
                    dataset[:, 0] = y_values
                    dataset[:, 1] = np.log(np.abs(y_values) + 1e-10)
                    dataset[:, 2 : 2 * num_var + 1 : 2] = x_values
                    dataset[:, 3 : 2 * num_var + 2 : 2] = np.log(
                        np.abs(x_values) + 1e-10
                    )
                else:
                    print("VARIABLE_TYPE should be one of 'normal', 'log', or 'both'")

                # save
                file_name = str(count_datasets).zfill(num_zfill)
                value_path = os.path.join(value_dir, f"data_{file_name}.npy")
                gt_path = os.path.join(gt_dir, f"data_{file_name}.npy")
                save_dataset(dataset, ground_truth, value_path, gt_path)
                count_datasets += 1
        except Exception as e:
            print(e)
            continue
    print(f"=> Number of unique expressions = {len(expr_uniq_seq)}")
    print(f"=> Number of datasets created = {count_datasets}")
    print("Finish!")


@hydra.main(version_base=None, config_path="./conf", config_name="transformer4sr.yaml")
def main(cfg: DictConfig):
    C, x1, x2, x3, x4, x5, x6 = sympy.symbols(
        "C, x1, x2, x3, x4, x5, x6", real=True, positive=True
    )
    generate_data(cfg)


if __name__ == "__main__":
    main()

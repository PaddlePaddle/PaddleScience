# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
Reference: https://github.com/meghbali/ANNElastoplasticity
"""

from os import path as osp

import functions
import hydra
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    (
        input_dict_train,
        label_dict_train,
        input_dict_val,
        label_dict_val,
    ) = functions.get_data(cfg.DATASET_STATE, cfg.DATASET_STRESS, cfg.NTRAIN_SIZE)
    model_list = functions.get_model_list(
        cfg.MODEL.ihlayers,
        cfg.MODEL.ineurons,
        input_dict_train["state_x"][0].shape[1],
        input_dict_train["state_y"][0].shape[1],
        input_dict_train["stress_x"][0].shape[1],
    )
    optimizer_list = functions.get_optimizer_list(
        model_list, cfg.TRAIN.epochs, cfg.TRAIN.iters_per_epoch
    )
    model_state1, model_state2, model_stress = model_list
    model_list_obj = ppsci.arch.ModelList(model_list)

    def transform_f(input, model, out_key):
        input11 = model(input)[out_key]
        input11 = input11.detach().clone()
        input_transformed = {}
        for key in input:
            input_transformed[key] = paddle.squeeze(input[key], axis=0)
        input1m = paddle.concat(
            x=(
                input11,
                paddle.index_select(
                    input_transformed["state_x"],
                    paddle.to_tensor([0, 1, 2, 3, 7, 8, 9, 10, 11, 12]),
                    axis=1,
                ),
            ),
            axis=1,
        )
        input_transformed["state_x_f"] = input1m
        return input_transformed

    def transform_f_stress(_in):
        return transform_f(_in, model_state1, "u")

    model_state1.register_input_transform(functions.transform_in)
    model_state2.register_input_transform(functions.transform_in)
    model_stress.register_input_transform(transform_f_stress)
    model_stress.register_output_transform(functions.transform_out)

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func),
        {
            "state_x": lambda out: out["state_x"],
            "state_y": lambda out: out["state_y"],
            "stress_x": lambda out: out["stress_x"],
            "stress_y": lambda out: out["stress_y"],
            "out_state1": lambda out: out["u"],
            "out_state2": lambda out: out["v"],
            "out_stress": lambda out: out["w"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.eval_loss_func),
        {
            "state_x": lambda out: out["state_x"],
            "state_y": lambda out: out["state_y"],
            "stress_x": lambda out: out["stress_x"],
            "stress_y": lambda out: out["stress_y"],
            "out_state1": lambda out: out["u"],
            "out_state2": lambda out: out["v"],
            "out_stress": lambda out: out["w"],
        },
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    functions.OUTPUT_DIR = cfg.output_dir
    # initialize solver
    solver = ppsci.solver.Solver(
        model_list_obj,
        constraint_pde,
        cfg.output_dir,
        optimizer_list,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        validator=validator_pde,
        eval_with_no_grad=cfg.TRAIN.eval_with_no_grad,
    )

    # train model
    solver.train()


def evaluate(cfg: DictConfig):
    print("Not supported.")


@hydra.main(version_base=None, config_path="./conf", config_name="epnn.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

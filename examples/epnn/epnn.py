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
    optimizer_list = functions.get_optimizer_list(model_list, cfg)
    model_state_elasto, model_state_plastic, model_stress = model_list
    model_list_obj = ppsci.arch.ModelList(model_list)

    def _transform_in_stress(_in):
        return functions.transform_in_stress(
            _in, model_state_elasto, "out_state_elasto"
        )

    model_state_elasto.register_input_transform(functions.transform_in)
    model_state_plastic.register_input_transform(functions.transform_in)
    model_stress.register_input_transform(_transform_in_stress)
    model_stress.register_output_transform(functions.transform_out)

    output_keys = [
        "state_x",
        "state_y",
        "stress_x",
        "stress_y",
        "out_state_elasto",
        "out_state_plastic",
        "out_stress",
    ]
    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
            "batch_size": 1,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.train_loss_func),
        {key: (lambda out, k=key: out[k]) for key in output_keys},
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
            "batch_size": 1,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.eval_loss_func),
        {key: (lambda out, k=key: out[k]) for key in output_keys},
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

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
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # train model
    solver.train()
    functions.plotting(cfg.output_dir)


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    (
        input_dict_train,
        _,
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
    model_state_elasto, model_state_plastic, model_stress = model_list
    model_list_obj = ppsci.arch.ModelList(model_list)

    def _transform_in_stress(_in):
        return functions.transform_in_stress(
            _in, model_state_elasto, "out_state_elasto"
        )

    model_state_elasto.register_input_transform(functions.transform_in)
    model_state_plastic.register_input_transform(functions.transform_in)
    model_stress.register_input_transform(_transform_in_stress)
    model_stress.register_output_transform(functions.transform_out)

    output_keys = [
        "state_x",
        "state_y",
        "stress_x",
        "stress_y",
        "out_state_elasto",
        "out_state_plastic",
        "out_stress",
    ]
    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
            "batch_size": 1,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(functions.eval_loss_func),
        {key: (lambda out, k=key: out[k]) for key in output_keys},
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}
    functions.OUTPUT_DIR = cfg.output_dir

    # initialize solver
    solver = ppsci.solver.Solver(
        model_list_obj,
        output_dir=cfg.output_dir,
        validator=validator_pde,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate
    solver.eval()


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

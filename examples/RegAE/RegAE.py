# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hydra
import paddle
from omegaconf import DictConfig
from paddle.nn import functional as F

import ppsci


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.AutoEncoder(**cfg.MODEL)

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "NPZDataset",
            "file_path": cfg.TRAIN_FILE_PATH,
            "input_keys": ("p_train",),
            "label_keys": ("p_train",),
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": False,
        },
    }

    def loss_expr(output_dict, label_dict, weight_dict=None):
        mu, log_sigma = output_dict["mu"], output_dict["log_sigma"]

        base = paddle.exp(2.0 * log_sigma) + paddle.pow(mu, 2) - 1.0 - 2.0 * log_sigma
        KLLoss = 0.5 * paddle.sum(base) / mu.shape[0]

        return {
            "decode_loss": F.mse_loss(output_dict["decoder_z"], label_dict["p_train"])
            + KLLoss
        }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(loss_expr),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NPZDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": ("p_train",),
            "label_keys": ("p_train",),
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(loss_expr),
        metric={"L2Rel": ppsci.metric.L2Rel()},
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.AutoEncoder(**cfg.MODEL)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NPZDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": ("p_train",),
            "label_keys": ("p_train",),
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.MSELoss(),
        output_expr={"p_hat": lambda out: out["p_hat"]},
        metric={"L2Rel": ppsci.metric.L2Rel()},
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    # evaluate after finished training
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="RegAE.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

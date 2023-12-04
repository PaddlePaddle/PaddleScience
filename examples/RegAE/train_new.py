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

from os import path as osp
from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import hydra
from omegaconf import DictConfig
from paddle import nn
from paddle.nn import functional as F

import ppsci
from ppsci.loss import KLLoss01
from ppsci.utils import logger

# from ppsci.utils import config

if TYPE_CHECKING:
    import paddle
    import pgl


criterion = nn.MSELoss()
kl_loss = KLLoss01()

# def train_mse_func(
#     output_dict: Dict[str, "paddle.Tensor"], label_dict: Dict[str, "pgl.Graph"], *args
# ) -> paddle.Tensor:
#     return F.mse_loss(output_dict["pred"], label_dict["label"].y)


def train_mse_func(
    # output_dict: Dict[str, "paddle.Tensor"], label_dict: Dict[str, "pgl.Graph"], *args
    mu,
    log_sigma,
    decoder_z,
    data_item,
) -> paddle.Tensor:
    # return F.mse_loss(output_dict["pred"], label_dict["label"].y)
    return kl_loss(mu, log_sigma) + criterion(decoder_z, data_item)


def eval_rmse_func(
    output_dict: Dict[str, List["paddle.Tensor"]],
    label_dict: Dict[str, List["pgl.Graph"]],
    *args,
) -> Dict[str, float]:
    mse_losses = [
        F.mse_loss(pred, label.y)
        for (pred, label) in zip(output_dict["pred"], label_dict["label"])
    ]
    return {"RMSE": (sum(mse_losses) / len(mse_losses)) ** 0.5}


def train(cfg: DictConfig):

    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.AutoEncoder(
        input_dim=cfg.MODEL.input_dim,
        latent_dim=cfg.MODEL.latent_dim,
        hidden_dim=cfg.MODEL.hidden_dim,
    )

    # set dataloader config
    ITERS_PER_EPOCH = 42
    train_dataloader_cfg = {
        "dataset": {
            "name": "VAECustomDataset",
            "file_path": "data/gaussian_train.npz",
            "data_type": "train",
        },
        "batch_size": 128,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        # output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(train_mse_func),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # # set validator
    # eval_dataloader_cfg = {
    #     "dataset": {
    #         "name": "VAECustomDataset",
    #         "file_path": "data/gaussian_train.npz",
    #         "data_type": "train",
    #     },
    #     "batch_size": 1,
    #     "sampler": {
    #         "name": "BatchSampler",
    #         "drop_last": False,
    #         "shuffle": False,
    #     },
    # }
    # rmse_validator = ppsci.validate.SupervisedValidator(
    #     eval_dataloader_cfg,
    #     loss=ppsci.loss.FunctionalLoss(train_mse_func),
    #     output_expr={"pred": lambda out: out["pred"]},
    #     metric={"RMSE": ppsci.metric.FunctionalMetric(eval_rmse_func)},
    #     name="RMSE_validator",
    # )
    # validator = {rmse_validator.name: rmse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        save_freq=50,
        eval_during_train=True,
        eval_freq=50,
        # validator=validator,
        eval_with_no_grad=True,
        # pretrained_model_path="./output_AMGNet/checkpoints/latest"
    )
    # train model
    solver.train()


@hydra.main(version_base=None, config_path="./conf", config_name="regae.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    # elif cfg.mode == "eval":
    #     evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

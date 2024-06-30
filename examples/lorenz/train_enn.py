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

# Two-stage training
# 1. Train a embedding model by running train_enn.py.
# 2. Load pretrained embedding model and freeze it, then train a transformer model by running train_transformer.py.

# This file is for step1: training a embedding model.
# This file is based on PaddleScience/ppsci API.
import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci


def get_mean_std(data: np.ndarray):
    mean = np.asarray(
        [np.mean(data[:, :, 0]), np.mean(data[:, :, 1]), np.mean(data[:, :, 2])]
    ).reshape(1, 3)
    std = np.asarray(
        [np.std(data[:, :, 0]), np.std(data[:, :, 1]), np.std(data[:, :, 2])]
    ).reshape(1, 3)
    return mean, std


def train(cfg: DictConfig):
    weights = (1.0 * (cfg.TRAIN_BLOCK_SIZE - 1), 1.0e4 * cfg.TRAIN_BLOCK_SIZE)
    regularization_key = "k_matrix"
    # manually build constraint(s)
    train_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": cfg.TRAIN_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "block_size": cfg.TRAIN_BLOCK_SIZE,
            "stride": 16,
            "weight_dict": {
                key: value for key, value in zip(cfg.MODEL.output_keys, weights)
            },
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 4,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELossWithL2Decay(
            regularization_dict={
                regularization_key: 1.0e-1 * (cfg.TRAIN_BLOCK_SIZE - 1)
            }
        ),
        {
            key: lambda out, k=key: out[k]
            for key in cfg.MODEL.output_keys + (regularization_key,)
        },
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # manually init model
    data_mean, data_std = get_mean_std(sup_constraint.data_loader.dataset.data)
    model = ppsci.arch.LorenzEmbedding(
        cfg.MODEL.input_keys,
        cfg.MODEL.output_keys + (regularization_key,),
        data_mean,
        data_std,
    )

    # init optimizer and lr scheduler
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.1)
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        iters_per_epoch=ITERS_PER_EPOCH,
        decay_steps=ITERS_PER_EPOCH,
        **cfg.TRAIN.lr_scheduler,
    )()
    optimizer = ppsci.optimizer.Adam(
        lr_scheduler, grad_clip=clip, **cfg.TRAIN.optimizer
    )(model)

    # manually build validator
    weights = (1.0 * (cfg.VALID_BLOCK_SIZE - 1), 1.0e4 * cfg.VALID_BLOCK_SIZE)
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "block_size": cfg.VALID_BLOCK_SIZE,
            "stride": 32,
            "weight_dict": {
                key: value for key, value in zip(cfg.MODEL.output_keys, weights)
            },
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": 4,
    }

    mse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        metric={"MSE": ppsci.metric.MSE()},
        name="MSE_Validator",
    )
    validator = {mse_validator.name: mse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=True,
        validator=validator,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    weights = (1.0 * (cfg.TRAIN_BLOCK_SIZE - 1), 1.0e4 * cfg.TRAIN_BLOCK_SIZE)
    regularization_key = "k_matrix"
    # manually build constraint(s)
    train_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": cfg.TRAIN_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "block_size": cfg.TRAIN_BLOCK_SIZE,
            "stride": 16,
            "weight_dict": {
                key: value for key, value in zip(cfg.MODEL.output_keys, weights)
            },
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 4,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELossWithL2Decay(
            regularization_dict={
                regularization_key: 1.0e-1 * (cfg.TRAIN_BLOCK_SIZE - 1)
            }
        ),
        {
            key: lambda out, k=key: out[k]
            for key in cfg.MODEL.output_keys + (regularization_key,)
        },
        name="Sup",
    )

    # manually init model
    data_mean, data_std = get_mean_std(sup_constraint.data_loader.dataset.data)
    model = ppsci.arch.LorenzEmbedding(
        cfg.MODEL.input_keys,
        cfg.MODEL.output_keys + (regularization_key,),
        data_mean,
        data_std,
    )

    # manually build validator
    weights = (1.0 * (cfg.VALID_BLOCK_SIZE - 1), 1.0e4 * cfg.VALID_BLOCK_SIZE)
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "block_size": cfg.VALID_BLOCK_SIZE,
            "stride": 32,
            "weight_dict": {
                key: value for key, value in zip(cfg.MODEL.output_keys, weights)
            },
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": 4,
    }

    mse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        metric={"MSE": ppsci.metric.MSE()},
        name="MSE_Validator",
    )
    validator = {mse_validator.name: mse_validator}

    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="enn.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

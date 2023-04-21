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
# 1. Train a embedding model by running train_enn_v2.py.
# 2. Load pretrained embedding model and freeze it, then train a transformer model by running train_transformer_v2.py.

# This file is for step1: training a embedding model.
# This file is based on PaddleScience/ppsci API.
import numpy as np
import paddle

import ppsci
from ppsci.utils import logger


def get_mean_std(data: np.ndarray):
    mean = np.asarray(
        [np.mean(data[:, :, 0]), np.mean(data[:, :, 1]), np.mean(data[:, :, 2])]
    ).reshape(1, 3)
    std = np.asarray(
        [np.std(data[:, :, 0]), np.std(data[:, :, 1]), np.std(data[:, :, 2])]
    ).reshape(1, 3)
    return mean, std


if __name__ == "__main__":
    ppsci.utils.set_random_seed(42)

    epochs = 300
    train_block_size = 16
    valid_block_size = 32

    input_keys = ("states",)
    output_keys = ("pred_states", "recover_states")
    weights = (1.0 * (train_block_size - 1), 1.0e4 * train_block_size)
    regularization_key = "k_matrix"

    output_dir = "./output/lorenz_enn"
    train_file_path = "./datasets/lorenz_training_rk.hdf5"
    valid_file_path = "./datasets/lorenz_valid_rk.hdf5"
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    # maunally build constraint(s)
    train_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": train_file_path,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "block_size": train_block_size,
            "stride": 16,
            "weight_dict": {key: value for key, value in zip(output_keys, weights)},
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 512,
        "num_workers": 4,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELossWithL2Decay(
            regularization_dict={regularization_key: 1.0e-1 * (train_block_size - 1)}
        ),
        {key: lambda out, k=key: out[k] for key in output_keys + (regularization_key,)},
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    iters_per_epoch = len(sup_constraint.data_loader)

    # manually init model
    data_mean, data_std = get_mean_std(sup_constraint.data_loader.dataset.data)
    model = ppsci.arch.LorenzEmbedding(
        input_keys, output_keys + (regularization_key,), data_mean, data_std
    )

    # init optimizer and lr scheduler
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.1)
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs,
        iters_per_epoch,
        0.001,
        gamma=0.995,
        decay_steps=iters_per_epoch,
        by_epoch=True,
    )()
    optimizer = ppsci.optimizer.Adam(
        lr_scheduler,
        weight_decay=1e-8,
        grad_clip=clip,
    )([model])

    # maunally build validator
    weights = (1.0 * (valid_block_size - 1), 1.0e4 * valid_block_size)
    eval_dataloader_cfg = {
        "dataset": {
            "name": "LorenzDataset",
            "file_path": valid_file_path,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "block_size": valid_block_size,
            "stride": 32,
            "weight_dict": {key: value for key, value in zip(output_keys, weights)},
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 512,
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
        output_dir,
        optimizer,
        lr_scheduler,
        epochs,
        iters_per_epoch,
        eval_during_train=True,
        validator=validator,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # directly evaluate pretrained model(optional)
    logger.init_logger("ppsci", f"{output_dir}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        output_dir=output_dir,
        validator=validator,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    solver.eval()

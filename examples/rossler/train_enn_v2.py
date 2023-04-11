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


def get_mean_std(data: np.ndarray):
    mean = np.asarray(
        [np.mean(data[:, :, 0]), np.mean(data[:, :, 1]), np.min(data[:, :, 2])]
    ).reshape(1, 3)
    std = np.asarray(
        [
            np.std(data[:, :, 0]),
            np.std(data[:, :, 1]),
            np.max(data[:, :, 2]) - np.min(data[:, :, 2]),
        ]
    ).reshape(1, 3)
    return mean, std


if __name__ == "__main__":
    ppsci.utils.set_random_seed(42)

    epochs = 300
    train_block_size = 16
    valid_block_size = 32

    input_keys = ["states"]
    output_keys = ["pred_states", "recover_states"]
    weights = [1.0 * (train_block_size - 1), 1.0e3 * train_block_size]
    regularization_key = "k_matrix"

    output_dir = "./output/rossler_enn"
    train_file_path = "/path/to/rossler_training.hdf5"
    valid_file_path = "/path/to/rossler_valid.hdf5"

    # maunally build constraint(s)
    train_dataloader = {
        "dataset": {
            "name": "RosslerDataset",
            "file_path": train_file_path,
            "block_size": train_block_size,
            "stride": 16,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 256,
        "num_workers": 4,
        "use_shared_memory": False,
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_file_path,
        input_keys,
        output_keys + [regularization_key],
        {},
        train_dataloader,
        ppsci.loss.MSELossWithL2Decay(
            regularization_dict={regularization_key: 1e-1 * (train_block_size - 1)}
        ),
        weight_dict={key: value for key, value in zip(output_keys, weights)},
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    iters_per_epoch = len(sup_constraint.data_loader)

    # manually init model
    mean, std = get_mean_std(sup_constraint.data_loader.dataset.data)
    model = ppsci.arch.RosslerEmbedding(
        input_keys, output_keys + [regularization_key], mean, std
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
    weights = [1.0 * (valid_block_size - 1), 1.0e4 * valid_block_size]
    eval_dataloader = {
        "dataset": {
            "name": "RosslerDataset",
            "file_path": valid_file_path,
            "block_size": valid_block_size,
            "stride": 32,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 8,
        "num_workers": 4,
        "use_shared_memory": False,
    }

    mse_metric = ppsci.validate.SupervisedValidator(
        input_keys,
        output_keys,
        eval_dataloader,
        ppsci.loss.MSELoss(),
        metric={"MSE": ppsci.metric.MSE()},
        weight_dict={key: value for key, value in zip(output_keys, weights)},
        name="MSE_Metric",
    )
    validator = {mse_metric.name: mse_metric}

    train_solver = ppsci.solver.Solver(
        "train",
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
    train_solver.train()

    eval_solver = ppsci.solver.Solver(
        "eval",
        model,
        constraint,
        output_dir,
        validator=validator,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    eval_solver.eval()

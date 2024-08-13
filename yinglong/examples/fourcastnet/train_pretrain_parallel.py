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

import os
import shutil

import numpy as np
import paddle.distributed as dist

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def copy_cur_file(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cur_file_path = os.path.abspath(__file__)
    dst_file_path = os.path.join(output_dir, os.path.basename(__file__))
    shutil.copy(cur_file_path, dst_file_path)


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(1024)
    # Initialize distributed environment
    dist.init_parallel_env()

    # set dataset path
    TRAIN_FILE_PATH = "../datasets/train"
    VALID_FILE_PATH = "../datasets/test"
    DATA_MEAN_PATH = "../datasets/stat/global_means.npy"
    DATA_STD_PATH = "../datasets/stat/global_stds.npy"
    DATA_TIME_MEAN_PATH = "../datasets/stat/time_means.npy"

    # set training hyper-parameters
    input_keys = ("input",)
    output_keys = ("output",)
    IMG_H, IMG_W = 720, 1440  # for HRRR dataset croped data
    EPOCHS = 60 if not args.epochs else args.epochs
    # FourCastNet HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
    # The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
    # 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500', 'v850', 'v1000',
    # 'mslp', 'u10', 'v10', 't2m'.
    VARS_CHANNEL = list(range(20))
    # set output directory
    OUTPUT_DIR = (
        "../output/era5_fourcastnet" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    copy_cur_file(OUTPUT_DIR)

    data_mean, data_std = fourcast_utils.get_mean_std(
        DATA_MEAN_PATH, DATA_STD_PATH, VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        DATA_TIME_MEAN_PATH, IMG_H, IMG_W, VARS_CHANNEL
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )
    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (IMG_H, IMG_W)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": TRAIN_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "transforms": transforms,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 1,
        "num_workers": 8,
    }
    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": VALID_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "transforms": transforms,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 8,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={
            "MAE": ppsci.metric.MAE(keep_batch=True),
            "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
                num_lat=IMG_H,
                std=data_std,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
            "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
                num_lat=IMG_H,
                mean=data_time_mean_normalize,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.AFNONet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL),
    )

    # init optimizer and lr scheduler
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        EPOCHS,
        ITERS_PER_EPOCH,
        5e-4,
        by_epoch=True,
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        validator=validator,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after finished training
    # solver.eval()

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
import warnings

import examples.fourcastnet_hrrr.utils as local_utils
import numpy as np
import paddle.distributed as dist

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

warnings.filterwarnings("ignore")


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
    TRAIN_FILE_PATH = "../train_data"
    VALID_FILE_PATH = "../test_data"
    DATA_MEAN_PATH = "../stat/mean_crop.npy"
    DATA_STD_PATH = "../stat/std_crop.npy"
    DATA_TIME_MEAN_PATH = "../stat/time_mean_crop.npy"

    USE_EXTRA = False
    if USE_EXTRA:
        EXTRA_FILE_PATH = "/root/ssd3/datasets/hrrr_h5_crop_rad/"
        EXTRA_DATA_MEAN_PATH = "/root/ssd3/datasets/hrrr_h5_crop_rad/stat/mean_crop.npy"
        EXTRA_DATA_STD_PATH = "/root/ssd3/datasets/hrrr_h5_crop_rad/stat/std_crop.npy"
        EXTRA_DATA_TIME_MEAN_PATH = (
            "/root/ssd3/datasets/hrrr_h5_crop_rad/stat/time_mean_crop.npy"
        )
        EXTRA_VARS_CHANNEL = list(range(2))
    else:
        EXTRA_FILE_PATH = None
        EXTRA_VARS_CHANNEL = None

    # set training hyper-parameters
    input_keys = ("input",)
    output_keys = ("output",)
    IMG_H, IMG_W = 440, 408  # for HRRR dataset croped data
    EPOCHS = 30 if not args.epochs else args.epochs
    # FourCastNet HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
    # The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
    # 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500', 'v850', 'v1000',
    # 'mslp', 'u10', 'v10', 't2m'.
    VARS_CHANNEL = list(range(24))
    # set output directory
    OUTPUT_DIR = (
        "../output/hrrr_time_embedding_merge_train"
        if not args.output_dir
        else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    copy_cur_file(OUTPUT_DIR)

    data_mean, data_std = local_utils.get_mean_std(
        DATA_MEAN_PATH, DATA_STD_PATH, VARS_CHANNEL
    )
    data_time_mean = local_utils.get_time_mean(
        DATA_TIME_MEAN_PATH, IMG_H, IMG_W, VARS_CHANNEL
    )

    if USE_EXTRA:
        extra_data_mean, extra_data_std = local_utils.get_mean_std(
            EXTRA_DATA_MEAN_PATH, EXTRA_DATA_STD_PATH, EXTRA_VARS_CHANNEL
        )
        extra_data_time_mean = local_utils.get_time_mean(
            EXTRA_DATA_TIME_MEAN_PATH, IMG_H, IMG_W, EXTRA_VARS_CHANNEL
        )
        data_mean = np.concatenate((data_mean, extra_data_mean))
        data_std = np.concatenate((data_std, extra_data_std))
        data_time_mean = np.concatenate((data_time_mean, extra_data_time_mean))

    data_time_mean_normalize = np.expand_dims(
        (data_time_mean - data_mean) / data_std, 0
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
            "name": "HRRRDataset",
            "file_path": TRAIN_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "transforms": transforms,
            "extra_file_path": EXTRA_FILE_PATH,
            "extra_vars_channel": EXTRA_VARS_CHANNEL,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 2,
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
            "name": "HRRRDataset",
            "file_path": VALID_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "transforms": transforms,
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
                std=data_std,
                keep_batch=True,
                variable_dict={"u10": 21, "v10": 22},
            ),
            "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
                mean=data_time_mean_normalize,
                keep_batch=True,
                variable_dict={"u10": 21, "v10": 22},
            ),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.AFNOAttnParallelUNet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL),
        attn_channel_ratio=[0.25] * 4 + [0.5] * 4 + [0.25] * 4,
    )

    model = ppsci.arch.AFNOAttnParallelNet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL) + len(EXTRA_VARS_CHANNEL)
        if USE_EXTRA
        else len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL) + len(EXTRA_VARS_CHANNEL)
        if USE_EXTRA
        else len(VARS_CHANNEL),
        attn_channel_ratio=0.25,
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
        # eval_during_train=True,
        # validator=validator,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()

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

import functools
from typing import Tuple

import h5py
import numpy as np
import paddle
import paddle.distributed as dist

import utils as local_utils
import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def get_vis_datas(
    file_path: str,
    date_strings: Tuple[str, ...],
    num_timestamps: int,
    vars_channel: Tuple[int, ...],
    img_h: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
):
    _file = h5py.File(file_path, "r")["fields"]
    data = []
    for date_str in date_strings:
        hours_since_jan_01_epoch = fourcast_utils.date_to_hours(date_str)
        ic = int(hours_since_jan_01_epoch / 6)
        data.append(_file[ic : ic + num_timestamps + 1, vars_channel, 0:img_h])
    data = np.asarray(data)

    vis_datas = {"input": (data[:, 0] - data_mean) / data_std}
    for t in range(num_timestamps):
        hour = (t + 1) * 6
        data_t = data[:, t + 1]
        wind_data = []
        for i in range(data_t.shape[0]):
            wind_data.append((data_t[i][0] ** 2 + data_t[i][1] ** 2) ** 0.5)
        vis_datas[f"target_{hour}h"] = np.asarray(wind_data)
    return vis_datas


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(1024)
    # Initialize distributed environment
    dist.init_parallel_env()

    # set dataset path
    TRAIN_FILE_PATH = "../../../ssd1/hrrr_data/train"
    VALID_FILE_PATH = "../../../ssd1/hrrr_data/valid"
    DATA_MEAN_PATH = "../../../ssd1/hrrr_data/stat/mean_crop.npy"
    DATA_STD_PATH = "../../../ssd1/hrrr_data/stat/std_crop.npy"
    DATA_TIME_MEAN_PATH = "../../../ssd1/hrrr_data/stat/time_mean_crop.npy"

    # set training hyper-parameters
    NUM_TIMESTAMPS = 6
    input_keys = ("input",)
    output_keys = tuple(f"output_{i}" for i in range(NUM_TIMESTAMPS))
    IMG_H, IMG_W = 440, 408
    EPOCHS = 15 if not args.epochs else args.epochs
    # FourCastNet HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
    # The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
    # 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500', 'v850', 'v1000',
    # 'mslp', 'u10', 'v10', 't2m'.
    VARS_CHANNEL = list(range(24))
    # set output directory
    OUTPUT_DIR =  "./output/finetune_6timestamp" if not args.output_dir else args.output_dir
    PRETRAINED_MODEL_PATH = "./output/finetune_1timestamp/checkpoints/latest"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    data_mean, data_std = local_utils.get_mean_std(
        DATA_MEAN_PATH, DATA_STD_PATH, VARS_CHANNEL
    )
    data_time_mean = local_utils.get_time_mean(
        DATA_TIME_MEAN_PATH, IMG_H, IMG_W, VARS_CHANNEL
    )
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
            "num_label_timestamps": NUM_TIMESTAMPS,
            "transforms": transforms,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": 4,
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
            "num_label_timestamps": NUM_TIMESTAMPS,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 8,
    }

    # set metirc
    metric = {
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
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric=metric,
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.AFNOAttnParallelNet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL),
      num_timestamps=NUM_TIMESTAMPS,
        attn_channel_ratio=0.25
    )

    # init optimizer and lr scheduler
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        EPOCHS,
        ITERS_PER_EPOCH,
        1e-4,
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
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after finished training
    # solver.eval()

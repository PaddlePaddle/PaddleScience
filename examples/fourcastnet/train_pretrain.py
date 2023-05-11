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

import numpy as np
import paddle.distributed as dist

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(1024)
    # Initialize distributed environment
    dist.init_parallel_env()

    # set dataset path
    TRAIN_FILE_PATH = "./datasets/era5/train"
    VALID_FILE_PATH = "./datasets/era5/test"
    DATA_MEAN_PATH = "./datasets/era5/stat/global_means.npy"
    DATA_STD_PATH = "./datasets/era5/stat/global_stds.npy"
    DATA_TIME_MEAN_PATH = "./datasets/era5/stat/time_means.npy"

    # set training hyper-parameters
    input_keys = ("input",)
    output_keys = ("output",)
    IMG_H, IMG_W = 720, 1440
    EPOCHS = 150 if not args.epochs else args.epochs
    # FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
    # The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
    # 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
    # You can obtain detailed information about each variable from
    # https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
    VARS_CHANNEL = list(range(20))
    # set output directory
    OUTPUT_DIR = (
        "./output/fourcastnet/pretrain" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

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
    USE_SAMPLED_DATA = False
    if not USE_SAMPLED_DATA:
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
    else:
        NUM_GPUS_PER_NODE = 8
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5SampledDataset",
                "file_path": TRAIN_FILE_PATH,
                "input_keys": input_keys,
                "label_keys": output_keys,
            },
            "sampler": {
                "name": "DistributedBatchSampler",
                "drop_last": True,
                "shuffle": True,
                "num_replicas": NUM_GPUS_PER_NODE,
                "rank": dist.get_rank() % NUM_GPUS_PER_NODE,
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
    model = ppsci.arch.AFNONet(input_keys, output_keys)

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
        log_freq=1,
        validator=validator,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # directly evaluate model from pretrained_model_path(optional)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        log_freq=1,
        validator=validator,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    solver.eval()

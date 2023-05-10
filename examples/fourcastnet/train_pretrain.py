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

import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def get_mean_std(mean_path, std_path, time_mean_path, vars_channel, img_h, img_w):
    mean = np.load(mean_path).squeeze(0).astype(np.float32)
    mean = mean[vars_channel]
    std = np.load(std_path).squeeze(0).astype(np.float32)
    std = std[vars_channel]
    time_mean = np.load(time_mean_path).astype(np.float32)
    time_mean = time_mean[:, vars_channel, :img_h, :img_w]
    return mean, std, time_mean


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(1024)
    # Initialize distributed environment
    dist.init_parallel_env()

    # set dataset path
    train_file_path = "./datasets/era5/train"
    valid_file_path = "./datasets/era5/test"
    data_mean_path = "./datasets/era5/stat/global_means.npy"
    data_std_path = "./datasets/era5/stat/global_stds.npy"
    data_time_mean_path = "./datasets/era5/stat/time_means.npy"

    # set training hyper-parameters
    input_keys = ["input"]
    output_keys = ["output"]
    img_h, img_w = 720, 1440
    epochs = 150 if not args.epochs else args.epochs
    # FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
    # The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
    # 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
    # You can obtain detailed information about each variable from
    # https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
    vars_channel = [i for i in range(20)]
    # set output directory
    output_dir = (
        "./output/fourcastnet/pretrain" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    data_mean, data_std, data_time_mean = get_mean_std(
        data_mean_path, data_std_path, data_time_mean_path, vars_channel, img_h, img_w
    )
    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (img_h, img_w)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]

    # set train dataloader config
    use_sampled_data = False
    if not use_sampled_data:
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5Dataset",
                "file_path": train_file_path,
                "input_keys": input_keys,
                "label_keys": output_keys,
                "vars_channel": vars_channel,
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
        num_gpus_per_node = 8
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5SampledDataset",
                "file_path": train_file_path,
                "input_keys": input_keys,
                "label_keys": output_keys,
            },
            "sampler": {
                "name": "DistributedBatchSampler",
                "drop_last": True,
                "shuffle": True,
                "num_replicas": num_gpus_per_node,
                "rank": dist.get_rank() % num_gpus_per_node,
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
    iters_per_epoch = len(sup_constraint.data_loader)

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": valid_file_path,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": vars_channel,
            "transforms": transforms,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 8,
        "num_workers": 0,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={
            "MAE": ppsci.metric.MAE(keep_batch=True),
            "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
                num_lat=img_h,
                std=data_std,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
            "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
                num_lat=img_h,
                mean=data_time_mean,
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
        epochs,
        iters_per_epoch,
        5e-4,
        by_epoch=True,
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

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
    logger.init_logger("ppsci", f"{output_dir}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        log_freq=1,
        validator=validator,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    solver.eval()

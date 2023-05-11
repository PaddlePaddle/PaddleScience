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

from functools import partial
from typing import Tuple

import h5py
import numpy as np
import paddle
import paddle.distributed as dist

import examples.fourcastnet.utils as fourcast_utils
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
    train_file_path = "./datasets/era5/train"
    valid_file_path = "./datasets/era5/test"
    test_file_path = "./datasets/era5/out_of_sample/2018.h5"
    data_mean_path = "./datasets/era5/stat/global_means.npy"
    data_std_path = "./datasets/era5/stat/global_stds.npy"
    data_time_mean_path = "./datasets/era5/stat/time_means.npy"

    # set training hyper-parameters
    num_timestamps = 2
    input_keys = ("input",)
    output_keys = tuple([f"output_{i}" for i in range(num_timestamps)])
    img_h, img_w = 720, 1440
    epochs = 50 if not args.epochs else args.epochs
    # FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
    # The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
    # 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
    # You can obtain detailed information about each variable from
    # https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
    vars_channel = list(range(20))
    # set output directory
    output_dir = (
        "./output/fourcastnet/finetune" if not args.output_dir else args.output_dir
    )
    pretrained_model_path = "./output/fourcastnet/pretrain/checkpoints/latest"
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    data_mean, data_std = fourcast_utils.get_mean_std(
        data_mean_path, data_std_path, vars_channel
    )
    data_time_mean = fourcast_utils.get_time_mean(
        data_time_mean_path, img_h, img_w, vars_channel
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )

    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (img_h, img_w)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]
    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": train_file_path,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": vars_channel,
            "num_label_timestamps": num_timestamps,
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
            "num_label_timestamps": num_timestamps,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": 1,
    }

    # set metirc
    metric = {
        "MAE": ppsci.metric.MAE(keep_batch=True),
        "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
            num_lat=img_h,
            std=data_std,
            keep_batch=True,
            variable_dict={"u10": 0, "v10": 1},
        ),
        "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
            num_lat=img_h,
            mean=data_time_mean_normalize,
            keep_batch=True,
            variable_dict={"u10": 0, "v10": 1},
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
    model = ppsci.arch.AFNONet(input_keys, output_keys, num_timestamps=num_timestamps)

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
        pretrained_model_path=pretrained_model_path,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # set testing hyper-parameters
    num_timestamps = 32
    output_keys = tuple([f"output_{i}" for i in range(num_timestamps)])

    # set model for testing
    model = ppsci.arch.AFNONet(input_keys, output_keys, num_timestamps=num_timestamps)

    # update eval dataloader config
    eval_dataloader_cfg["dataset"].update(
        {
            "file_path": test_file_path,
            "label_keys": output_keys,
            "num_label_timestamps": num_timestamps,
            "stride": 8,
        }
    )

    # set validator for testing
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric=metric,
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer datas
    date_strings = ("2018-09-08 00:00:00",)
    vis_datas = get_vis_datas(
        test_file_path,
        date_strings,
        num_timestamps,
        vars_channel,
        img_h,
        data_mean,
        data_std,
    )

    def output_wind_func(d, var_name, data_mean, data_std):
        output = (d[var_name] * data_std) + data_mean
        wind_data = []
        for i in range(output.shape[0]):
            wind_data.append((output[i][0] ** 2 + output[i][1] ** 2) ** 0.5)
        return paddle.to_tensor(wind_data)

    vis_output_expr = {}
    for i in range(num_timestamps):
        hour = (i + 1) * 6
        vis_output_expr[f"output_{hour}h"] = partial(
            output_wind_func,
            var_name=f"output_{i}",
            data_mean=paddle.to_tensor(data_mean),
            data_std=paddle.to_tensor(data_std),
        )
        vis_output_expr[f"target_{hour}h"] = lambda d, hour=hour: d[f"target_{hour}h"]
    # set visualizer
    visualizer = {
        "visulize_wind": ppsci.visualize.VisualizerWeather(
            vis_datas,
            vis_output_expr,
            xticks=np.linspace(0, 1439, 13),
            xticklabels=[str(i) for i in range(360, -1, -30)],
            yticks=np.linspace(0, 719, 7),
            yticklabels=[str(i) for i in range(90, -91, -30)],
            vmin=0,
            vmax=25,
            colorbar_label="m\s",
            batch_size=1,
            num_timestamps=num_timestamps,
            prefix="wind",
        )
    }

    # directly evaluate pretrained model
    logger.init_logger("ppsci", f"{output_dir}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        output_dir=output_dir,
        log_freq=1,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    solver.eval()
    # visualize prediction from pretrained_model_path
    solver.visualize()

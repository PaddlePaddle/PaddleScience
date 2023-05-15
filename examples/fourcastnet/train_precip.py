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

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def get_vis_datas(
    wind_file_path: str,
    file_path: str,
    date_strings: Tuple[str, ...],
    num_timestamps: int,
    vars_channel: Tuple[int, ...],
    img_h: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
):
    __wind_file = h5py.File(wind_file_path, "r")["fields"]
    _file = h5py.File(file_path, "r")["tp"]
    wind_data = []
    data = []
    for date_str in date_strings:
        hours_since_jan_01_epoch = fourcast_utils.date_to_hours(date_str)
        ic = int(hours_since_jan_01_epoch / 6)
        wind_data.append(__wind_file[ic, vars_channel, 0:img_h])
        data.append(_file[ic + 1 : ic + num_timestamps + 1, 0:img_h])
    wind_data = np.asarray(wind_data)
    data = np.asarray(data)

    vis_datas = {"input": (wind_data - data_mean) / data_std}
    for t in range(num_timestamps):
        hour = (t + 1) * 6
        data_t = data[:, t]
        vis_datas[f"target_{hour}h"] = np.asarray(data_t)
    return vis_datas


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(1024)
    # Initialize distributed environment
    dist.init_parallel_env()

    # set wind dataset path
    WIND_TRAIN_FILE_PATH = "./datasets/era5/train"
    WIND_VALID_FILE_PATH = "./datasets/era5/test"
    WIND_TEST_FILE_PATH = "./datasets/era5/out_of_sample/2018.h5"
    WIND_MEAN_PATH = "./datasets/era5/stat/global_means.npy"
    WIND_STD_PATH = "./datasets/era5/stat/global_stds.npy"
    WIND_TIME_MEAN_PATH = "./datasets/era5/stat/time_means.npy"
    # set dataset path
    TRAIN_FILE_PATH = "./datasets/era5/precip/train"
    VALID_FILE_PATH = "./datasets/era5/precip/test"
    TEST_FILE_PATH = "./datasets/era5/precip/out_of_sample/2018.h5"
    TIME_MEAN_PATH = "./datasets/era5/stat/precip/time_means.npy"

    # set training hyper-parameters
    input_keys = ("input",)
    output_keys = ("output",)
    IMG_H, IMG_W = 720, 1440
    EPOCHS = 25 if not args.epochs else args.epochs
    # FourCastNet use 20 atmospheric variable，their index in the dataset is from 0 to 19.
    # The variable name is 'u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z000',
    # 'u850', 'v850', 'z850',  'u500', 'v500', 'z500', 't500', 'z50', 'r500', 'r850', 'tcwv'.
    # You can obtain detailed information about each variable from
    # https://cds.climate.copernicus.eu/cdsapp#!/search?text=era5&type=dataset
    VARS_CHANNEL = list(range(20))
    # set output directory
    OUTPUT_DIR = (
        "./output/fourcastnet/precip" if not args.output_dir else args.output_dir
    )
    WIND_MODEL_PATH = "./output/fourcastnet/finetune/checkpoints/latest"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    wind_data_mean, wind_data_std = fourcast_utils.get_mean_std(
        WIND_MEAN_PATH, WIND_STD_PATH, VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(TIME_MEAN_PATH, IMG_H, IMG_W)

    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (IMG_H, IMG_W)}},
        {
            "Normalize": {
                "mean": wind_data_mean,
                "std": wind_data_std,
                "apply_keys": ("input",),
            }
        },
        {"Log1p": {"scale": 1e-5, "apply_keys": ("label",)}},
    ]

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": WIND_TRAIN_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "precip_file_path": TRAIN_FILE_PATH,
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
            "file_path": WIND_VALID_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "precip_file_path": VALID_FILE_PATH,
            "transforms": transforms,
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
            num_lat=IMG_H, keep_batch=True, unlog=True
        ),
        "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
            num_lat=IMG_H, mean=data_time_mean, keep_batch=True, unlog=True
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
    wind_model = ppsci.arch.AFNONet(input_keys, output_keys)
    ppsci.utils.save_load.load_pretrain(wind_model, path=WIND_MODEL_PATH)
    model = ppsci.arch.PrecipNet(input_keys, output_keys, wind_model=wind_model)

    # init optimizer and lr scheduler
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        EPOCHS,
        ITERS_PER_EPOCH,
        2.5e-4,
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
    solver.eval()

    # set testing hyper-parameters
    NUM_TIMESTAMPS = 6
    output_keys = tuple([f"output_{i}" for i in range(NUM_TIMESTAMPS)])

    # set model for testing
    model = ppsci.arch.PrecipNet(
        input_keys, output_keys, num_timestamps=NUM_TIMESTAMPS, wind_model=wind_model
    )

    # update eval dataloader config
    eval_dataloader_cfg["dataset"].update(
        {
            "file_path": WIND_TEST_FILE_PATH,
            "label_keys": output_keys,
            "precip_file_path": TEST_FILE_PATH,
            "num_label_timestamps": NUM_TIMESTAMPS,
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

    # set set visualizer datas
    DATE_STRINGS = ("2018-04-04 00:00:00",)
    vis_datas = get_vis_datas(
        WIND_TEST_FILE_PATH,
        TEST_FILE_PATH,
        DATE_STRINGS,
        NUM_TIMESTAMPS,
        VARS_CHANNEL,
        IMG_H,
        wind_data_mean,
        wind_data_std,
    )

    def output_precip_func(d, var_name):
        output = 1e-2 * paddle.expm1(d[var_name][0])
        return output

    visu_output_expr = {}
    for i in range(NUM_TIMESTAMPS):
        hour = (i + 1) * 6
        visu_output_expr[f"output_{hour}h"] = functools.partial(
            output_precip_func,
            var_name=f"output_{i}",
        )
        visu_output_expr[f"target_{hour}h"] = (
            lambda d, hour=hour: d[f"target_{hour}h"] * 1000
        )
    # set visualizer
    visualizer = {
        "visulize_precip": ppsci.visualize.VisualizerWeather(
            vis_datas,
            visu_output_expr,
            xticks=np.linspace(0, 1439, 13),
            xticklabels=[str(i) for i in range(360, -1, -30)],
            yticks=np.linspace(0, 719, 7),
            yticklabels=[str(i) for i in range(90, -91, -30)],
            vmin=0.001,
            vmax=130,
            colorbar_label="mm",
            log_norm=True,
            batch_size=1,
            num_timestamps=NUM_TIMESTAMPS,
            prefix="precip",
        )
    }

    # directly evaluate pretrained model
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        output_dir=OUTPUT_DIR,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    solver.eval()
    # visualize prediction from pretrained_model_path
    solver.visualize()

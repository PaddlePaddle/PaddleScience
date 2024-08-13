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
import visualdl as vdl

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
    TRAIN_FILE_PATH = "../datasets/era5_tiny/train"
    TEST_FILE_PATH = "../datasets/era5_tiny/out_of_sample"
    DATA_MEAN_PATH = "../datasets/era5_tiny/stat/global_means.npy"
    DATA_STD_PATH = "../datasets/era5_tiny/stat/global_stds.npy"
    DATA_TIME_MEAN_PATH = "../datasets/era5_tiny/stat/time_means.npy"

    # set training hyper-parameters
    NUM_TIMESTAMPS = 40
    input_keys = ("input",)
    output_keys = tuple(f"output_{i}" for i in range(NUM_TIMESTAMPS))
    IMG_H, IMG_W = 720, 1440
    # FourCastNet HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
    # The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
    # 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500', 'v850', 'v1000',
    # 'mslp', 'u10', 'v10', 't2m'.
    VARS_CHANNEL = list(range(20))
    VARIABLE_DICT = {
        "u10": 0,
        "v10": 1,
        "t2m": 2,
        "sp": 3,
        "msl": 4,
        "t850": 5,
        "u1000": 6,
        "v1000": 7,
        "z1000": 8,
        "u850": 9,
        "v850": 10,
        "z850": 11,
        "u500": 12,
        "v500": 13,
        "z500": 14,
        "t500": 15,
        "z50": 16,
        "r500": 17,
        "r850": 18,
        "tcwv": 19,
    }
    # set output directory
    OUTPUT_DIR = (
        "../output/era5_parallel_unet_025_05_025"
        if not args.output_dir
        else args.output_dir
    )
    PRETRAINED_MODEL_PATH = "../output/era5_parallel_unet_025_05_025/checkpoints/latest"
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/infer.log", "info")

    vdl_writer = vdl.LogWriter(f"{OUTPUT_DIR}/vdl_no_weight")

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

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": TEST_FILE_PATH,
            "input_keys": input_keys,
            "label_keys": output_keys,
            "vars_channel": VARS_CHANNEL,
            "transforms": transforms,
            "num_label_timestamps": NUM_TIMESTAMPS,
            "training": False,
            "stride": 8,
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
            num_lat=IMG_H,
            std=data_std,
            keep_batch=True,
            variable_dict=VARIABLE_DICT,
        ),
        "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
            num_lat=IMG_H,
            mean=data_time_mean_normalize,
            keep_batch=True,
            variable_dict=VARIABLE_DICT,
        ),
    }

    # set model
    model = ppsci.arch.AFNOAttnParallelUNet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL),
        attn_channel_ratio=[0.25] * 4 + [0.5] * 4 + [0.25] * 4,
        num_timestamps=NUM_TIMESTAMPS,
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
    # DATE_STRINGS = ("2018-09-08 00:00:00",)
    # vis_datas = get_vis_datas(
    #     VALID_FILE_PATH,
    #     DATE_STRINGS,
    #     NUM_TIMESTAMPS,
    #     VARS_CHANNEL,
    #     IMG_H,
    #     data_mean,
    #     data_std,
    # )

    # def output_wind_func(d, var_name, data_mean, data_std):
    #     output = (d[var_name] * data_std) + data_mean
    #     wind_data = []
    #     for i in range(output.shape[0]):
    #         wind_data.append((output[i][0] ** 2 + output[i][1] ** 2) ** 0.5)
    #     return paddle.to_tensor(wind_data, paddle.get_default_dtype())

    # vis_output_expr = {}
    # for i in range(NUM_TIMESTAMPS):
    #     hour = (i + 1) * 6
    #     vis_output_expr[f"output_{hour}h"] = functools.partial(
    #         output_wind_func,
    #         var_name=f"output_{i}",
    #         data_mean=paddle.to_tensor(data_mean, paddle.get_default_dtype()),
    #         data_std=paddle.to_tensor(data_std, paddle.get_default_dtype()),
    #     )
    #     vis_output_expr[f"target_{hour}h"] = lambda d, hour=hour: d[f"target_{hour}h"]
    # # set visualizer
    # visualizer = {
    #     "visulize_wind": ppsci.visualize.VisualizerWeather(
    #         vis_datas,
    #         vis_output_expr,
    #         xticks=np.linspace(0, 1439, 13),
    #         xticklabels=[str(i) for i in range(360, -1, -30)],
    #         yticks=np.linspace(0, 719, 7),
    #         yticklabels=[str(i) for i in range(90, -91, -30)],
    #         vmin=0,
    #         vmax=25,
    #         colorbar_label="m\s",
    #         batch_size=1,
    #         num_timestamps=NUM_TIMESTAMPS,
    #         prefix="wind",
    #     )
    # }

    # directly evaluate pretrained model
    solver = ppsci.solver.Solver(
        model,
        output_dir=OUTPUT_DIR,
        validator=validator,
        # visualizer=visualizer,
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
        vdl_writer=vdl_writer,
    )
    solver.eval()
    # visualize prediction from pretrained_model_path
    # solver.visualize()

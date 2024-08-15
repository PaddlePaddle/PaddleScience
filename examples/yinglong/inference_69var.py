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
import os
import shutil
from typing import Tuple

import h5py
import numpy as np
import paddle
import paddle.distributed as dist
import visualdl as vdl

import examples.fourcastnet_hrrr.utils as local_utils
import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def copy_cur_file(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cur_file_path = os.path.abspath(__file__)
    dst_file_path = os.path.join(output_dir, os.path.basename(__file__))
    shutil.copy(cur_file_path, dst_file_path)


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
    TRAIN_FILE_PATH = "../train_data"
    VALID_FILE_PATH = "/root/ssd3/datasets/hrrr_h5_crop_69var/valid"
    DATA_MEAN_PATH = "/root/ssd3/datasets/hrrr_h5_crop_69var/stat/mean_crop.npy"
    DATA_STD_PATH = "/root/ssd3/datasets/hrrr_h5_crop_69var/stat/std_crop.npy"
    DATA_TIME_MEAN_PATH = (
        "/root/ssd3/datasets/hrrr_h5_crop_69var/stat/time_mean_crop.npy"
    )

    # set training hyper-parameters
    NUM_TIMESTAMPS = 48
    input_keys = ("input",)
    output_keys = tuple(f"output_{i}" for i in range(NUM_TIMESTAMPS))
    IMG_H, IMG_W = 440, 408
    # FourCastNet HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
    # The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
    # 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500', 'v850', 'v1000',
    # 'mslp', 'u10', 'v10', 't2m'.
    VARS_CHANNEL = list(range(69))
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
    var_names = [
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "s50",
        "s100",
        "s150",
        "s200",
        "s250",
        "s300",
        "s400",
        "s500",
        "s600",
        "s700",
        "s850",
        "s925",
        "s1000",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "mslp",
        "u10",
        "v10",
        "t2m",
    ]
    VARIABLE_DICT = {name: i for i, name in enumerate(var_names)}
    # set output directory
    # N = 2 if not args.num_timestamps else args.num_timestamps
    # OUTPUT_DIR = (
    #     f"../output/hrrr_parallel_7years_ratio025_1011_finetune_{N}"
    #     if not args.output_dir
    #     else args.output_dir
    # )
    # PRETRAINED_MODEL_PATH = f"../output/hrrr_parallel_7years_ratio025_1011_finetune_{N}/checkpoints/best_model"

    OUTPUT_DIR = (
        "../output/hrrr_time_embedding_69var_depth24"
        if not args.output_dir
        else args.output_dir
    )
    PRETRAINED_MODEL_PATH = (
        f"../output/hrrr_time_embedding_69var_depth24/checkpoints/latest"
    )

    copy_cur_file(OUTPUT_DIR)
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/infer.log", "info")

    vdl_writer = vdl.LogWriter(f"{OUTPUT_DIR}/vdl_no_weight")

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
            "stride": 24,
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
            std=data_std,
            keep_batch=True,
            variable_dict=VARIABLE_DICT,
        ),
        "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
            mean=data_time_mean_normalize,
            keep_batch=True,
            variable_dict=VARIABLE_DICT,
        ),
    }

    # set model
    model = ppsci.arch.AFNOAttnParallelNet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL),
        num_timestamps=NUM_TIMESTAMPS,
        attn_channel_ratio=0.25,
        depth=24,
        embed_dim=1024,
        num_blocks=16,
        num_heads=16,
    )
    import pdb

    pdb.set_trace()
    flops = paddle.flops(model, [1, 69, 440, 408], print_detail=True)
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

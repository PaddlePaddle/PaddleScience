# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import path as osp

import h5py
import hydra
import numpy as np
import paddle
import pandas as pd
from omegaconf import DictConfig
from packaging import version

from examples.yinglong.plot import save_plot_weather_from_dict
from examples.yinglong.predictor import YingLongPredictor
from ppsci.utils import logger


def inference(cfg: DictConfig):
    # log paddlepaddle's version
    if version.Version(paddle.__version__) != version.Version("0.0.0"):
        paddle_version = paddle.__version__
        if version.Version(paddle.__version__) < version.Version("2.6.0"):
            logger.warning(
                f"Detected paddlepaddle version is '{paddle_version}', "
                "currently it is recommended to use release 2.6 or develop version."
            )
    else:
        paddle_version = f"develop({paddle.version.commit[:7]})"

    logger.info(f"Using paddlepaddle {paddle_version}")

    num_timestamps = cfg.INFER.num_timestamps
    # create predictor
    predictor = YingLongPredictor(cfg)

    # load data
    # HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
    # The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
    # 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500',
    # 'v850', 'v1000', 'mslp', 'u10', 'v10', 't2m'.
    input_file = h5py.File(cfg.INFER.input_file, "r")["fields"]
    nwp_file = h5py.File(cfg.INFER.nwp_file, "r")["fields"]

    # input_data.shape: (1, 24, 440, 408)
    input_data = input_file[0:1]
    # nwp_data.shape: # (num_timestamps, 24, 440, 408)
    nwp_data = nwp_file[0:num_timestamps]
    # ground_truth.shape: (num_timestamps, 24, 440, 408)
    ground_truth = input_file[1 : num_timestamps + 1]

    # create time stamps
    cur_time = pd.to_datetime(cfg.INFER.init_time, format="%Y/%m/%d/%H")
    time_stamps = [[cur_time]]
    for _ in range(num_timestamps):
        cur_time += pd.Timedelta(hours=1)
        time_stamps.append([cur_time])

    # run predictor
    pred_data = predictor.predict(input_data, time_stamps, nwp_data)
    pred_data = pred_data.squeeze(axis=1)  # (num_timestamps, 24, 440, 408)

    # save predict data
    save_path = osp.join(cfg.output_dir, "result.npy")
    np.save(save_path, pred_data)
    logger.info(f"Save output to {save_path}")

    # plot wind data
    u10_idx, v10_idx = 21, 22
    pred_wind = (pred_data[:, u10_idx] ** 2 + pred_data[:, v10_idx] ** 2) ** 0.5
    ground_truth_wind = (
        ground_truth[:, u10_idx] ** 2 + ground_truth[:, v10_idx] ** 2
    ) ** 0.5
    data_dict = {}
    visu_keys = []
    for i in range(num_timestamps):
        visu_key = f"Init time: {cfg.INFER.init_time}h\n Ground truth: {i+1}h"
        visu_keys.append(visu_key)
        data_dict[visu_key] = ground_truth_wind[i]
        visu_key = f"Init time: {cfg.INFER.init_time}h\n YingLong-12 Layers: {i+1}h"
        visu_keys.append(visu_key)
        data_dict[visu_key] = pred_wind[i]

    save_plot_weather_from_dict(
        foldername=cfg.output_dir,
        data_dict=data_dict,
        visu_keys=visu_keys,
        xticks=np.linspace(0, 407, 7),
        xticklabels=[str(i) for i in range(0, 409, 68)],
        yticks=np.linspace(0, 439, 9),
        yticklabels=[str(i) for i in range(0, 441, 55)],
        vmin=0,
        vmax=15,
        colorbar_label="m/s",
        num_timestamps=12,  # only plot 12 timestamps
    )
    logger.info(f"Save plot to {cfg.output_dir}")


@hydra.main(version_base=None, config_path="./conf", config_name="yinglong_12.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

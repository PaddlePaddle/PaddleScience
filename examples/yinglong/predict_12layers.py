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

import argparse
from os import path as osp

import h5py
import numpy as np
import paddle
import paddle.inference as paddle_infer
import pandas as pd
from packaging import version

from examples.yinglong.timefeatures import time_features
from ppsci.utils import logger


class YingLong:
    def __init__(
        self, model_file: str, params_file: str, mean_path: str, std_path: str
    ):
        self.model_file = model_file
        self.params_file = params_file

        config = paddle_infer.Config(model_file, params_file)
        config.switch_ir_optim(False)
        config.enable_use_gpu(100, 0)
        config.enable_memory_optim()

        self.predictor = paddle_infer.create_predictor(config)

        # get input names and data handles
        self.input_names = self.predictor.get_input_names()
        self.input_data_handle = self.predictor.get_input_handle(self.input_names[0])
        self.time_stamps_handle = self.predictor.get_input_handle(self.input_names[1])
        self.nwp_data_handle = self.predictor.get_input_handle(self.input_names[2])

        # get output names and data handles
        self.output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])

        # load mean and std data
        self.mean = np.load(mean_path).reshape(-1, 1, 1).astype(np.float32)
        self.std = np.load(std_path).reshape(-1, 1, 1).astype(np.float32)

    def _preprocess_data(self, input_data, time_stamps, nwp_data):
        # normalize data
        input_data = (input_data - self.mean) / self.std
        nwp_data = (nwp_data - self.mean) / self.std

        # process time stamps
        for i in range(len(time_stamps)):
            time_stamps[i] = pd.DataFrame({"date": time_stamps[i]})
            time_stamps[i] = time_features(time_stamps[i], timeenc=1, freq="h").astype(
                np.float32
            )
        time_stamps = np.asarray(time_stamps)
        return input_data, time_stamps, nwp_data

    def _postprocess_data(self, data):
        # denormalize data
        data = data * self.std + self.mean
        return data

    def __call__(self, input_data, time_stamp, nwp_data):
        # preprocess data
        input_data, time_stamps, nwp_data = self._preprocess_data(
            input_data, time_stamp, nwp_data
        )

        # set input data
        self.input_data_handle.copy_from_cpu(input_data)
        self.time_stamps_handle.copy_from_cpu(time_stamps)
        self.nwp_data_handle.copy_from_cpu(nwp_data)

        # run predictor
        self.predictor.run()

        # get output data
        output_data = self.output_handle.copy_to_cpu()

        # postprocess data
        output_data = self._postprocess_data(output_data)
        return output_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./yinglong_models/yinglong_12.pdmodel",
        help="Model filename",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./yinglong_models/yinglong_12.pdiparams",
        help="Parameter filename",
    )
    parser.add_argument(
        "--mean_path",
        type=str,
        default="./hrrr_example_24vars/stat/mean_crop.npy",
        help="Mean filename",
    )
    parser.add_argument(
        "--std_path",
        type=str,
        default="./hrrr_example_24vars/stat/std_crop.npy",
        help="Standard deviation filename",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./hrrr_example_24vars/valid/2022/01/01.h5",
        help="Input filename",
    )
    parser.add_argument(
        "--init_time", type=str, default="2022/01/01/00", help="Init time"
    )
    parser.add_argument(
        "--nwp_file",
        type=str,
        default="./hrrr_example_24vars/nwp_convert/2022/01/01/00.h5",
        help="NWP filename",
    )
    parser.add_argument(
        "--num_timestamps", type=int, default=22, help="Number of timestamps"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Output file path"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.init_logger("ppsci", osp.join(args.output_path, "predict.log"), "info")
    if version.Version(paddle.__version__) != version.Version("2.5.2"):
        logger.error(
            f"Your Paddle version is {paddle.__version__}, but this code currently "
            "only supports PaddlePaddle 2.5.2. The latest version of Paddle will be "
            "supported as soon as possible."
        )
        exit()

    num_timestamps = args.num_timestamps

    # create predictor
    predictor = YingLong(
        args.model_file, args.params_file, args.mean_path, args.std_path
    )

    # load data
    input_file = h5py.File(args.input_file, "r")["fields"]
    nwp_file = h5py.File(args.nwp_file, "r")["fields"]
    input_data = input_file[0:1]
    nwp_data = nwp_file[0:num_timestamps]

    # create time stamps
    cur_time = pd.to_datetime(args.init_time, format="%Y/%m/%d/%H")
    time_stamps = [[cur_time]]
    for _ in range(num_timestamps):
        cur_time += pd.Timedelta(hours=1)
        time_stamps.append([cur_time])

    # run predictor
    output_data = predictor(input_data, time_stamps, nwp_data)

    save_path = osp.join(args.output_path, "result.npy")
    logger.info(f"Save output to {save_path}")
    np.save(save_path, output_data)


if __name__ == "__main__":
    main()

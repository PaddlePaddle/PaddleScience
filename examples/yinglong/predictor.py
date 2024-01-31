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


import numpy as np
import paddle.inference as paddle_infer
import pandas as pd

from examples.yinglong.timefeatures import time_features


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

        # get predict data
        pred_data = self.output_handle.copy_to_cpu()

        # postprocess data
        pred_data = self._postprocess_data(pred_data)
        return pred_data

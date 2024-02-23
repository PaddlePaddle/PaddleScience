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


from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from deploy.python_infer import base
from examples.yinglong.timefeatures import time_features
from ppsci.utils import logger


class YingLongPredictor(base.Predictor):
    """General predictor for YingLong model.

    Args:
        cfg (DictConfig): Running configuration.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(
            cfg.INFER.pdmodel_path,
            cfg.INFER.pdpiparams_path,
            device=cfg.INFER.device,
            engine=cfg.INFER.engine,
            precision=cfg.INFER.precision,
            onnx_path=cfg.INFER.onnx_path,
            ir_optim=cfg.INFER.ir_optim,
            min_subgraph_size=cfg.INFER.min_subgraph_size,
            gpu_mem=cfg.INFER.gpu_mem,
            gpu_id=cfg.INFER.gpu_id,
            max_batch_size=cfg.INFER.max_batch_size,
            num_cpu_threads=cfg.INFER.num_cpu_threads,
        )
        self.log_freq = cfg.log_freq

        # get input names and data handles
        self.input_names = self.predictor.get_input_names()
        self.input_data_handle = self.predictor.get_input_handle(self.input_names[0])
        self.time_stamps_handle = self.predictor.get_input_handle(self.input_names[1])
        self.nwp_data_handle = self.predictor.get_input_handle(self.input_names[2])

        # get output names and data handles
        self.output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])

        # load mean and std data
        self.mean = np.load(cfg.INFER.mean_path).reshape(-1, 1, 1).astype("float32")
        self.std = np.load(cfg.INFER.std_path).reshape(-1, 1, 1).astype("float32")

    def _preprocess_data(
        self,
        input_data: np.ndarray,
        time_stamps: List[List[pd.Timestamp]],
        nwp_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _postprocess_data(self, data: np.ndarray):
        # denormalize data
        data = data * self.std + self.mean
        return data

    def predict(
        self,
        input_data: np.ndarray,
        time_stamps: List[List[pd.Timestamp]],
        nwp_data: np.ndarray,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Predicts the output of the yinglong model for the given input.

        Args:
            input_data (np.ndarray): Input data of shape (N, T, H, W).
            time_stamps (List[List[pd.Timestamp]]): Timestamps data.
            nwp_data (np.ndarray): NWP data.
            batch_size (int, optional): Batch size, now only support 1. Defaults to 1.

        Returns:
            np.ndarray: Prediction.
        """
        if batch_size != 1:
            raise ValueError(
                f"YingLongPredictor only support batch_size=1, but got {batch_size}"
            )

        # prepare input handle(s)
        input_handles = {
            self.input_names[0]: self.input_data_handle,
            self.input_names[1]: self.time_stamps_handle,
            self.input_names[2]: self.nwp_data_handle,
        }
        # prepare output handle(s)
        output_handles = {self.output_names[0]: self.output_handle}

        num_samples = len(input_data)
        if num_samples != 1:
            raise ValueError(
                f"YingLongPredictor only support num_samples=1, but got {num_samples}"
            )

        batch_num = 1

        # inference by batch
        for batch_id in range(1, batch_num + 1):
            if batch_id % self.log_freq == 0 or batch_id == batch_num:
                logger.info(f"Predicting batch {batch_id}/{batch_num}")

            # preprocess data
            input_data, time_stamps, nwp_data = self._preprocess_data(
                input_data, time_stamps, nwp_data
            )
            # prepare batch input dict
            batch_input_dict = {
                self.input_names[0]: input_data,
                self.input_names[1]: time_stamps,
                self.input_names[2]: nwp_data,
            }

            # send batch input data to input handle(s)
            for name, handle in input_handles.items():
                handle.copy_from_cpu(batch_input_dict[name])

            # run predictor
            self.predictor.run()

            # receive batch output data from output handle(s)
            pred = output_handles[self.output_names[0]].copy_to_cpu()
            pred = self._postprocess_data(pred)

        return pred

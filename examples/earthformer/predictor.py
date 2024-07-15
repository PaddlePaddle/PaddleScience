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
from omegaconf import DictConfig

from deploy.python_infer import base


class EarthformerPredictor(base.Predictor):
    """General predictor for Earthformer model.

    Args:
        cfg (DictConfig): Running configuration.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__(
            cfg.INFER.pdmodel_path,
            cfg.INFER.pdiparams_path,
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

        # get output names and data handles
        self.output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])

    def predict(
        self,
        input_data: np.ndarray,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Predicts the output of the yinglong model for the given input.

        Args:
            input_data (np.ndarray): Input data of shape (N, T, H, W).
            batch_size (int, optional): Batch size, now only support 1. Defaults to 1.
        Returns:
            np.ndarray: Prediction.
        """
        if batch_size != 1:
            raise ValueError(
                f"EarthformerPredictor only support batch_size=1, but got {batch_size}"
            )
        # prepare input handle(s)
        input_handles = {self.input_names[0]: self.input_data_handle}
        # prepare output handle(s)
        output_handles = {self.output_names[0]: self.output_handle}

        # prepare batch input dict
        batch_input_dict = {
            self.input_names[0]: input_data,
        }
        # send batch input data to input handle(s)
        for name, handle in input_handles.items():
            handle.copy_from_cpu(batch_input_dict[name])

        # run predictor
        self.predictor.run()

        # receive batch output data from output handle(s)
        pred = output_handles[self.output_names[0]].copy_to_cpu()

        return pred

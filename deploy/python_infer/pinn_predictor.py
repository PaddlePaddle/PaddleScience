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

from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import paddle
from typing_extensions import Literal

from deploy.python_infer import base
from ppsci.utils import logger
from ppsci.utils import misc


class PINNPredictor(base.Predictor):
    def __init__(
        self,
        pdmodel_path: Optional[str] = None,
        pdpiparams_path: Optional[str] = None,
        *,
        device: Literal["gpu", "cpu", "npu", "xpu"] = "cpu",
        engine: Literal["native", "tensorrt", "onnx", "mkldnn"] = "native",
        precision: Literal["fp32", "fp16", "int8"] = "fp32",
        onnx_path: Optional[str] = None,
        ir_optim: bool = True,
        min_subgraph_size: int = 15,
        gpu_mem: int = 500,
        gpu_id: int = 0,
        max_batch_size: int = 10,
        num_cpu_threads: int = 10,
    ):
        super().__init__(
            pdmodel_path,
            pdpiparams_path,
            device=device,
            engine=engine,
            precision=precision,
            onnx_path=onnx_path,
            ir_optim=ir_optim,
            min_subgraph_size=min_subgraph_size,
            gpu_mem=gpu_mem,
            gpu_id=gpu_id,
            max_batch_size=max_batch_size,
            num_cpu_threads=num_cpu_threads,
        )

    def predict(
        self,
        input_dict: Dict[str, Union[np.ndarray, paddle.Tensor]],
        batch_size: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts the output of the model for the given input.

        Args:
            input_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]):
                A dictionary containing the input data.
            batch_size (int, optional): The batch size to use for prediction.
                Defaults to 64.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the predicted output.
        """
        if batch_size > self.max_batch_size:
            logger.warning(
                f"batch_size({batch_size}) is larger than "
                f"max_batch_size({self.max_batch_size}), which may occur error."
            )

        # prepare input handle(s)
        input_handles = {
            name: self.predictor.get_input_handle(name) for name in input_dict
        }
        # prepare output handle(s)
        output_handles = {
            name: self.predictor.get_output_handle(name)
            for name in self.predictor.get_output_names()
        }

        num_samples = len(next(iter(input_dict.values())))
        batch_num = (num_samples + (batch_size - 1)) // batch_size
        pred_dict = misc.Prettydefaultdict(list)

        for batch_id in range(batch_num):
            if batch_id % 10 == 0:
                logger.info(f"Predicting batch {batch_id}/{batch_num}")
            batch_input_dict = {}
            st = batch_id * batch_size
            ed = min(num_samples, (batch_id + 1) * batch_size)

            # prepare batch input dict
            for key in input_dict:
                batch_input_dict[key] = input_dict[key][st:ed]

            # send batch input data to input handle(s)
            for name, handle in input_handles.items():
                handle.copy_from_cpu(batch_input_dict[name])

            # run predictor
            self.predictor.run()

            # receive batch output data from output handle(s)
            batch_output_dict = {
                name: output_handles[name].copy_to_cpu() for name in output_handles
            }

            # collect batch data
            for key, batch_output in batch_output_dict.items():
                pred_dict[key].append(batch_output)

        # concatenate local predictions
        pred_dict = {key: np.concatenate(value) for key, value in pred_dict.items()}
        return pred_dict

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
from typing import List
from typing import Union

import numpy as np
import paddle
from omegaconf import DictConfig

from deploy.python_infer import base
from ppsci.utils import logger
from ppsci.utils import misc


class PINNPredictor(base.Predictor):
    """General predictor for PINN-based models.

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

        if self.engine != "onnx":
            # prepare input handle(s)
            input_handles = {
                name: self.predictor.get_input_handle(name) for name in input_dict
            }
            # prepare output handle(s)
            output_handles = {
                name: self.predictor.get_output_handle(name)
                for name in self.predictor.get_output_names()
            }
        else:
            # input_names = [node_arg.name for node_arg in self.predictor.get_inputs()]
            output_names: List[str] = [
                node_arg.name for node_arg in self.predictor.get_outputs()
            ]

        num_samples = len(next(iter(input_dict.values())))
        batch_num = (num_samples + (batch_size - 1)) // batch_size
        pred_dict = misc.Prettydefaultdict(list)

        # inference by batch
        for batch_id in range(1, batch_num + 1):
            if batch_id % self.log_freq == 0 or batch_id == batch_num:
                logger.info(f"Predicting batch {batch_id}/{batch_num}")

            # prepare batch input dict
            st = (batch_id - 1) * batch_size
            ed = min(num_samples, batch_id * batch_size)
            batch_input_dict = {key: input_dict[key][st:ed] for key in input_dict}

            # send batch input data to input handle(s)
            if self.engine != "onnx":
                for name, handle in input_handles.items():
                    handle.copy_from_cpu(batch_input_dict[name])

            # run predictor
            if self.engine != "onnx":
                self.predictor.run()
                # receive batch output data from output handle(s)
                batch_output_dict = {
                    name: output_handles[name].copy_to_cpu() for name in output_handles
                }
            else:
                batch_outputs = self.predictor.run(
                    output_names=output_names,
                    input_feed=batch_input_dict,
                )
                batch_output_dict = {
                    name: output for (name, output) in zip(output_names, batch_outputs)
                }

            # collect batch output data
            for key, batch_output in batch_output_dict.items():
                pred_dict[key].append(batch_output)

        # concatenate local predictions
        pred_dict = {key: np.concatenate(value) for key, value in pred_dict.items()}

        return pred_dict

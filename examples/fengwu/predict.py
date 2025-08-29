# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
from typing import List

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from packaging import version

from deploy.python_infer import base
from ppsci.utils import logger


class FengWuPredictor(base.Predictor):
    """General predictor for FengWu model.

    Args:
        cfg (DictConfig): Running configuration.
    """

    # 14 day with time-interval of six hours
    PREDICT_TIMESTAMP = int(14 * 24 / 6)
    # Where 69 represents 69 atmospheric features, The first four variables are surface variables in the order of ['u10', 'v10', 't2m', 'msl'],
    # followed by non-surface variables in the order of ['z', 'q', 'u', 'v', 't']. Each data has 13 levels, which are ordered as
    # [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000].
    # Therefore, the order of the 69 variables is [u10, v10, t2m, msl, z50, z100, ..., z1000, q50, q100, ..., q1000, t50, t100, ..., t1000].
    NUM_ATMOSPHERIC_FEATURES = 69

    def __init__(
        self,
        cfg: DictConfig,
    ):
        assert cfg.INFER.engine == "onnx", "FengWu engine only supports 'onnx'."

        super().__init__(
            pdmodel_path=None,
            pdiparams_path=None,
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

        # get input names
        self.input_names = [
            input_node.name for input_node in self.predictor.get_inputs()
        ]

        # get output names
        self.output_names = [
            output_node.name for output_node in self.predictor.get_outputs()
        ]

        # load mean and std data
        self.data_mean = np.load(cfg.INFER.mean_path)[:, np.newaxis, np.newaxis]
        self.data_std = np.load(cfg.INFER.std_path)[:, np.newaxis, np.newaxis]

    def _preprocess_data(
        self, input_data_prev: np.ndarray, input_data_next: np.ndarray
    ) -> np.ndarray:
        input_data_prev_after_norm = (
            input_data_prev.astype("float32") - self.data_mean
        ) / self.data_std
        input_data_next_after_norm = (
            input_data_next.astype("float32") - self.data_mean
        ) / self.data_std
        input_data = np.concatenate(
            (input_data_prev_after_norm, input_data_next_after_norm), axis=0
        )[np.newaxis, :, :, :]
        input_data = input_data.astype(np.float32)

        return input_data

    def predict(
        self,
        input_data_prev: np.ndarray,
        input_data_next: np.ndarray,
        batch_size: int = 1,
    ) -> List[np.ndarray]:
        """Predicts the output of the yinglong model for the given input.

        Args:
            input_data_prev(np.ndarray): Atomospheric data at the first time moment.
            input_data_next(np.ndarray): Atmospheric data six later.
            batch_size (int, optional): Batch size, now only support 1. Defaults to 1.

        Returns:
            List[np.ndarray]: Prediction for next 56 hours.
        """
        if batch_size != 1:
            raise ValueError(
                f"FengWuPredictor only support batch_size=1, but got {batch_size}"
            )

        # process data
        input_data = self._preprocess_data(input_data_prev, input_data_next)

        output_data_list = []
        # prepare input dict
        for _ in range(self.PREDICT_TIMESTAMP):
            input_dict = {
                self.input_names[0]: input_data,
            }

            # run predictor
            output_data = self.predictor.run(None, input_dict)[0]
            input_data = np.concatenate(
                (
                    input_data[:, self.NUM_ATMOSPHERIC_FEATURES :],
                    output_data[:, : self.NUM_ATMOSPHERIC_FEATURES],
                ),
                axis=1,
            )
            output_data = (
                output_data[0, : self.NUM_ATMOSPHERIC_FEATURES] * self.data_std
            ) + self.data_mean

            output_data_list.append(output_data)

        return output_data_list


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

    # create predictor
    predictor = FengWuPredictor(cfg)

    # load data
    input_data_prev = np.load(cfg.INFER.input_file).astype(np.float32)
    input_data_next = np.load(cfg.INFER.input_next_file).astype(np.float32)

    # run predictor
    output_data_list = predictor.predict(input_data_prev, input_data_next)

    # save predict data
    for i in range(FengWuPredictor.PREDICT_TIMESTAMP):
        output_save_path = osp.join(cfg.output_dir, f"output_{i}.npy")
        np.save(output_save_path, output_data_list[i])
        logger.info(f"Save output with timestamp:{i} to {output_save_path}.")


@hydra.main(version_base=None, config_path="./conf", config_name="fengwu.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

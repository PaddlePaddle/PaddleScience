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
from typing import Tuple

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from packaging import version

from deploy.python_infer import base
from ppsci.utils import logger


class PanguWeatherPredictor(base.Predictor):
    """General predictor for PanguWeather model.

    Args:
        cfg (DictConfig): Running configuration.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        assert cfg.INFER.engine == "onnx", "Pangu-Weather engine only supports 'onnx'."

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

    def predict(
        self,
        input_data: np.ndarray,
        input_surface_data: np.ndarray,
        batch_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts the output of the yinglong model for the given input.

        Args:
            input_data (np.ndarray): Input data.
            input_surface_data (np.ndarray): Input Surface data.
            batch_size (int, optional): Batch size, now only support 1. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prediction.
        """
        if batch_size != 1:
            raise ValueError(
                f"PanguWeatherPredictor only support batch_size=1, but got {batch_size}"
            )

        # prepare input dict
        input_dict = {
            self.input_names[0]: input_data,
            self.input_names[1]: input_surface_data,
        }

        # run predictor
        output_data, output_surface_data = self.predictor.run(None, input_dict)

        return output_data, output_surface_data


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
    predictor = PanguWeatherPredictor(cfg)

    # load data
    input_data = np.load(cfg.INFER.input_file).astype(np.float32)
    input_surface_data = np.load(cfg.INFER.input_surface_file).astype(np.float32)

    # run predictor
    output_data, output_surface_data = predictor.predict(input_data, input_surface_data)

    # save predict data
    output_save_path = osp.join(cfg.output_dir, "output_upper.npy")
    np.save(output_save_path, output_data)
    output_surface_save_path = osp.join(cfg.output_dir, "output_surface.npy")
    np.save(output_surface_save_path, output_surface_data)
    logger.info(
        f"Save output upper to {output_save_path} and output surface to {output_surface_save_path}."
    )


@hydra.main(version_base=None, config_path="./conf", config_name="pangu_weather.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

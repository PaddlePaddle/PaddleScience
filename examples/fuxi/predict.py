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

import hydra
import numpy as np
import paddle
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from omegaconf import OmegaConf
from packaging import version
from util import save_like

from deploy.python_infer import base
from ppsci.utils import logger


def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, "H") for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)


class FuXiPredictor(base.Predictor):
    """General predictor for FuXi model.

    Args:
        cfg (DictConfig): Running configuration.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        print(f"cfg: {cfg}")
        assert cfg.INFER.engine == "onnx", "FuXi engine only supports 'onnx'."

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

        self.output_dir = cfg.output_dir

    def predict(
        self, input_data, tembs, global_step, stage, num_step, data, batch_size: int = 1
    ) -> tuple[np.ndarray, int]:
        """Predicts the output of the yinglong model for the given input.

        Args:
            input_data(np.ndarray): Atomospheric data of two preceding time steps
            tembs(np.ndarray): Encoded timestamp.
            global_step (int): The global step of forecast.
            stage (int): The stage of forecast model.
            num_step (int): The number of forecast steps.
            batch_size (int, optional): Batch size, now only support 1. Defaults to 1.

        Returns:
            tuple[np.ndarray, int]: Prediction for one stage and the global step.
        """
        if batch_size != 1:
            raise ValueError(
                f"FuXiPredictor only support batch_size=1, but got {batch_size}"
            )

        # prepare input dict
        for _ in range(0, num_step):
            input_dict = {
                self.input_names[0]: input_data,
                self.input_names[1]: tembs[global_step],
            }

            # run predictor
            new_input = self.predictor.run(None, input_dict)[0]
            output = new_input[:, -1]
            save_like(output, data, global_step, self.output_dir)
            print(
                f"stage: {stage}, global_step: {global_step+1:02d}, output: {output.min():.2f} {output.max():.2f}"
            )
            input_data = new_input
            global_step += 1

        return input_data, global_step


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

    num_steps = cfg.num_steps
    stages = ["short", "medium", "long"]

    # load data
    data = xr.open_dataarray(cfg.input_file)

    total_step = sum(num_steps)
    init_time = pd.to_datetime(data.time.values[-1])
    tembs = time_encoding(init_time, total_step)

    print(f'init_time: {init_time.strftime(("%Y%m%d-%H"))}')
    print(f"latitude: {data.lat.values[0]} ~ {data.lat.values[-1]}")

    assert data.lat.values[0] == 90
    assert data.lat.values[-1] == -90

    input_data = data.values[None]

    step = 0
    for i, num_step in enumerate(num_steps):
        print(f"Inference {stages[i]} ...")
        cfg_path = cfg.fuxi_config_dir + "fuxi_" + stages[i] + ".yaml"
        config = OmegaConf.load(cfg_path)
        print(f"predictor_cfg: {config}")
        predictor = FuXiPredictor(config)
        # run predictor
        input_data, step = predictor.predict(
            input_data=input_data,
            tembs=tembs,
            global_step=step,
            stage=i,
            num_step=num_step,
            data=data,
        )

        if step > total_step:
            break


@hydra.main(version_base=None, config_path="./conf", config_name="fuxi.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

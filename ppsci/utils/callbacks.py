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
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from ppsci.utils import logger
from ppsci.utils import misc


class InitCallback(Callback):
    """Callback class for:
    1. Fixing random seed to 'config.seed'
    2. Initialize logger while creating output directory(if not exist).

    NOTE: This callback is mainly for reducing unnecessary duplicate code in each
    examples code when runing with hydra.

    This callback should be added to hydra config file as follows:

    ``` yaml hl_lines="7-11"
    # content of example.yaml below
    hydra:
      run:
        ...
      job:
        ...
      callbacks:
        init_callback:
          _target_: ppsci.utils.callbacks.InitCallback # <-- add callback at here
        xxx_callback:
          _target_: ppsci.utils.callbacks.XxxCallback # <-- add more callback here
      sweep:
          ...
    ...
    ...
    ```
    """

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        # fix random seed for reproducibility
        misc.set_random_seed(config.seed)

        # create output directory
        logger.init_logger(
            "ppsci", osp.join(config.output_dir, f"{config.mode}.log"), "info"
        )

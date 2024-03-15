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

import importlib.util
import inspect
import sys
from os import path as osp
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from ppsci.utils import config as config_module
from ppsci.utils import logger
from ppsci.utils import misc

RUNTIME_EXIT_CODE = 1  # for other errors
VALIDATION_ERROR_EXIT_CODE = 2  # for invalid argument detected in config file


class InitCallback(Callback):
    """Callback class for:
    1. Parse config dict from given yaml file and check its validity, complete missing items by its' default values.
    2. Fixing random seed to 'config.seed'.
    3. Initialize logger while creating output directory(if not exist).

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
        # check given cfg using pre-defined pydantic schema in 'SolverConfig', error(s) will be raised
        # if any checking failed at this step
        if importlib.util.find_spec("pydantic") is not None:
            from pydantic import ValidationError
        else:
            logger.error(
                f"ModuleNotFoundError at {__file__}:{inspect.currentframe().f_lineno}\n"
                "Please install pydantic with `pip install pydantic` when set callbacks"
                " in your config yaml."
            )
            sys.exit(RUNTIME_EXIT_CODE)

        # check given cfg using pre-defined pydantic schema in 'SolverConfig',
        # error(s) will be printed and exit program if any checking failed at this step
        try:
            _model_pydantic = config_module.SolverConfig(**dict(config))
            # complete missing items with default values pre-defined in pydantic schema in
            # 'SolverConfig'
            full_cfg = DictConfig(_model_pydantic.model_dump())
        except ValidationError as e:
            print(e)
            sys.exit(VALIDATION_ERROR_EXIT_CODE)
        except Exception as e:
            print(e)
            sys.exit(RUNTIME_EXIT_CODE)

        # fix random seed for reproducibility
        misc.set_random_seed(full_cfg.seed)

        # initialze logger while creating output directory
        logger.init_logger(
            "ppsci",
            osp.join(full_cfg.output_dir, f"{full_cfg.mode}.log")
            if full_cfg.output_dir and full_cfg.mode not in ["export", "infer"]
            else None,
            full_cfg.log_level,
        )

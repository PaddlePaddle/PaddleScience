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
import os
import sys
import traceback
from os import path as osp
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from ppsci.utils import config as config_module
from ppsci.utils import logger
from ppsci.utils import misc

RUNTIME_EXIT_CODE = 1  # for other errors
VALIDATION_ERROR_EXIT_CODE = 2  # for invalid argument detected in config file


class InitCallback(Callback):
    """Callback class for:
    1. Parse config dict from given yaml file and check its validity.
    2. Fixing random seed to 'config.seed'.
    3. Initialize logger while creating output directory(if not exist).
    4. Enable prim mode if specified.

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
            full_cfg = DictConfig(_model_pydantic.model_dump())
        except ValidationError as e:
            print(e)
            sys.exit(VALIDATION_ERROR_EXIT_CODE)
        except Exception as e:
            print(e)
            sys.exit(RUNTIME_EXIT_CODE)

        # fix random seed for reproducibility
        misc.set_random_seed(full_cfg.seed)

        # initialize logger while creating output directory
        logger.init_logger(
            "ppsci",
            osp.join(full_cfg.output_dir, f"{full_cfg.mode}.log")
            if full_cfg.output_dir and full_cfg.mode not in ["export", "infer"]
            else None,
            full_cfg.log_level,
        )

        # set device before running into example function
        if "device" in full_cfg:
            import paddle

            if isinstance(full_cfg.device, str):
                paddle.device.set_device(full_cfg.device)

        try:
            if "num" in HydraConfig.get().job:
                jobs_id = HydraConfig.get().job.num
            else:
                jobs_id = None
            if "n_jobs" in HydraConfig.get().launcher:
                parallel_jobs_num = HydraConfig.get().launcher.n_jobs
            else:
                parallel_jobs_num = None

            if jobs_id and parallel_jobs_num:
                job_device_id = jobs_id % parallel_jobs_num
                device_type = paddle.get_device().split(":")[0]
                logger.message(
                    f"Running job {jobs_id} on device {device_type}:{job_device_id}(logical device id)"
                )
                paddle.set_device(f"{device_type}:{job_device_id}")
        except Exception as e:
            print(e)
            traceback.print_exc()
            sys.exit(RUNTIME_EXIT_CODE)

        # enable prim if specified
        if "prim" in full_cfg and bool(full_cfg.prim):
            # Mostly for compiler running with dy2st.
            from paddle.framework import core

            core.set_prim_eager_enabled(True)
            core._set_prim_all_enabled(True)
            logger.message("Prim mode is enabled.")

        # === Optionally log git info & dump uncommitted diff ===
        if bool(full_cfg.get("trace", False)):
            if not importlib.util.find_spec("git"):
                logger.error(
                    "[Code Trace] GitPython is required for trace=True.\n"
                    "Please install it with: pip install GitPython"
                )
                sys.exit(RUNTIME_EXIT_CODE)

            from git import InvalidGitRepositoryError
            from git import Repo

            try:
                repo = Repo(".", search_parent_directories=True)
                branch = repo.active_branch.name
                commit = repo.head.commit
                commit_hash = commit.hexsha
                commit_time = commit.committed_datetime.isoformat()
                is_dirty = repo.is_dirty()

                logger.message("[Code Trace] Git Information:")
                logger.message(f"  Branch : {branch}")
                logger.message(f"  Commit : {commit_hash}")
                logger.message(f"  Date   : {commit_time}")
                logger.message(f"  Dirty  : {is_dirty}")

                if is_dirty:
                    trace_dir = osp.join(full_cfg.output_dir, "code_snapshot")
                    os.makedirs(trace_dir, exist_ok=True)

                    staged_diff = repo.git.diff("--cached")
                    if len(staged_diff) > 0:
                        staged_diff_path = osp.join(trace_dir, "staged.diff")
                        with open(staged_diff_path, "w", encoding="utf-8") as f:
                            f.write(staged_diff)
                        logger.info(
                            f"[Code Trace] Staged changes saved to: {staged_diff_path}"
                        )
                        logger.info(
                            f"[Code Trace] To restore your code to this staged version, run: git apply {staged_diff_path}"
                        )

                    unstaged_diff = repo.git.diff()
                    if len(unstaged_diff) > 0:
                        unstaged_diff_path = osp.join(trace_dir, "unstaged.diff")
                        with open(unstaged_diff_path, "w", encoding="utf-8") as f:
                            f.write(unstaged_diff)
                        logger.info(
                            f"[Code Trace] Unstaged changes saved to: {unstaged_diff_path}"
                        )
                        logger.info(
                            f"[Code Trace] To restore your code to this unstaged version, run: git apply {unstaged_diff_path}"
                        )

            except InvalidGitRepositoryError:
                logger.warning("[Code Trace] Not a Git repository. Skipping.")
            except Exception as e:
                logger.warning(f"[Code Trace] Unexpected error: {e}")

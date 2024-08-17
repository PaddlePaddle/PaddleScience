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

from __future__ import annotations

import importlib.util
from typing import Mapping
from typing import Optional
from typing import Tuple

from typing_extensions import Literal

__all__ = []

if importlib.util.find_spec("pydantic") is not None:
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from pydantic import BaseModel
    from pydantic import field_validator
    from pydantic import model_validator
    from pydantic_core.core_schema import ValidationInfo

    __all__.append("SolverConfig")

    class EMAConfig(BaseModel):
        use_ema: bool = False
        decay: float = 0.9
        avg_freq: int = 1

        @field_validator("decay")
        def decay_check(cls, v):
            if v <= 0 or v >= 1:
                raise ValueError(
                    f"'ema.decay' should be in (0, 1) when is type of float, but got {v}"
                )
            return v

        @field_validator("avg_freq")
        def avg_freq_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'ema.avg_freq' should be a positive integer when is type of int, "
                    f"but got {v}"
                )
            return v

    class SWAConfig(BaseModel):
        use_swa: bool = False
        avg_freq: int = 1
        avg_range: Optional[Tuple[int, int]] = None

        @field_validator("avg_range")
        def avg_range_check(cls, v, info: ValidationInfo):
            if isinstance(v, tuple) and v[0] > v[1]:
                raise ValueError(
                    f"'swa.avg_range' should be a valid range, but got {v}."
                )
            if isinstance(v, tuple) and v[0] < 0:
                raise ValueError(
                    "The start epoch of 'swa.avg_range' should be a non-negtive integer"
                    f" , but got {v[0]}."
                )
            if isinstance(v, tuple) and v[1] > info.data["epochs"]:
                raise ValueError(
                    "The end epoch of 'swa.avg_range' should not be lager than "
                    f"'epochs'({info.data['epochs']}), but got {v[1]}."
                )
            return v

        @field_validator("avg_freq")
        def avg_freq_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'swa.avg_freq' should be a positive integer when is type of int, "
                    f"but got {v}"
                )
            return v

    class TrainConfig(BaseModel):
        """
        Schema of training config for pydantic validation.
        """

        epochs: int = 0
        iters_per_epoch: int = 20
        update_freq: int = 1
        save_freq: int = 0
        eval_during_train: bool = False
        start_eval_epoch: int = 1
        eval_freq: int = 1
        checkpoint_path: Optional[str] = None
        pretrained_model_path: Optional[str] = None
        ema: Optional[EMAConfig] = None
        swa: Optional[SWAConfig] = None

        # Fine-grained validator(s) below
        @field_validator("epochs")
        def epochs_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'TRAIN.epochs' should be a positive integer when is type of int, "
                    f"but got {v}"
                )
            return v

        @field_validator("iters_per_epoch")
        def iters_per_epoch_check(cls, v):
            if v <= 0 and v != -1:
                raise ValueError(
                    f"'TRAIN.iters_per_epoch' received an invalid value({v}), "
                    "but is expected one of: \n"
                    "* A positive integer, to manually specify the number of iterations per epoch, "
                    "which is commonly used in PINN training.\n"
                    "* -1, to automatically set the number of iterations per epoch to "
                    "the length of dataloader of given constraint, which is commonly "
                    f"used in data-driven training.\n"
                )
            return v

        @field_validator("update_freq")
        def update_freq_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'TRAIN.update_freq' should be a positive integer when is type of int"
                    f", but got {v}"
                )
            return v

        @field_validator("save_freq")
        def save_freq_check(cls, v):
            if v < 0:
                raise ValueError(
                    "'TRAIN.save_freq' should be a non-negtive integer when is type of int"
                    f", but got {v}"
                )
            return v

        @field_validator("start_eval_epoch")
        def start_eval_epoch_check(cls, v, info: ValidationInfo):
            if info.data["eval_during_train"]:
                if v <= 0:
                    raise ValueError(
                        f"'TRAIN.start_eval_epoch' should be a positive integer when "
                        f"'TRAIN.eval_during_train' is True, but got {v}"
                    )
            return v

        @field_validator("eval_freq")
        def eval_freq_check(cls, v, info: ValidationInfo):
            if info.data["eval_during_train"]:
                if v <= 0:
                    raise ValueError(
                        f"'TRAIN.eval_freq' should be a positive integer when "
                        f"'TRAIN.eval_during_train' is True, but got {v}"
                    )
            return v

        @model_validator(mode="after")
        def ema_swa_checker(self):
            if (self.ema and self.swa) and (self.ema.use_ema and self.swa.use_swa):
                raise ValueError(
                    "Cannot enable both EMA and SWA at the same time, "
                    "please disable at least one of them."
                )
            return self

    class EvalConfig(BaseModel):
        """
        Schema of evaluation config for pydantic validation.
        """

        pretrained_model_path: Optional[str] = None
        eval_with_no_grad: bool = False
        compute_metric_by_batch: bool = False
        batch_size: Optional[int] = 256

        @field_validator("batch_size")
        def batch_size_check(cls, v):
            if isinstance(v, int) and v <= 0:
                raise ValueError(
                    f"'EVAL.batch_size' should be greater than 0 or None, but got {v}"
                )
            return v

    class InferConfig(BaseModel):
        """
        Schema of inference config for pydantic validation.
        """

        pretrained_model_path: Optional[str] = None
        export_path: str = "./inference"
        pdmodel_path: Optional[str] = None
        pdiparams_path: Optional[str] = None
        onnx_path: Optional[str] = None
        device: Literal["gpu", "cpu", "npu", "xpu"] = "cpu"
        engine: Literal["native", "tensorrt", "onnx", "mkldnn"] = "native"
        precision: Literal["fp32", "fp16", "int8"] = "fp32"
        ir_optim: bool = True
        min_subgraph_size: int = 30
        gpu_mem: int = 2000
        gpu_id: int = 0
        max_batch_size: int = 1024
        num_cpu_threads: int = 10
        batch_size: Optional[int] = 256

        # Fine-grained validator(s) below
        @field_validator("engine")
        def engine_check(cls, v, info: ValidationInfo):
            if v == "tensorrt" and info.data["device"] != "gpu":
                raise ValueError(
                    "'INFER.device' should be 'gpu' when 'INFER.engine' is 'tensorrt', "
                    f"but got '{info.data['device']}'"
                )
            if v == "mkldnn" and info.data["device"] != "cpu":
                raise ValueError(
                    "'INFER.device' should be 'cpu' when 'INFER.engine' is 'mkldnn', "
                    f"but got '{info.data['device']}'"
                )

            return v

        @field_validator("min_subgraph_size")
        def min_subgraph_size_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'INFER.min_subgraph_size' should be greater than 0, "
                    f"but got {v}"
                )
            return v

        @field_validator("gpu_mem")
        def gpu_mem_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'INFER.gpu_mem' should be greater than 0, " f"but got {v}"
                )
            return v

        @field_validator("gpu_id")
        def gpu_id_check(cls, v):
            if v < 0:
                raise ValueError(
                    "'INFER.gpu_id' should be greater than or equal to 0, "
                    f"but got {v}"
                )
            return v

        @field_validator("max_batch_size")
        def max_batch_size_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'INFER.max_batch_size' should be greater than 0, " f"but got {v}"
                )
            return v

        @field_validator("num_cpu_threads")
        def num_cpu_threads_check(cls, v):
            if v < 0:
                raise ValueError(
                    "'INFER.num_cpu_threads' should be greater than or equal to 0, "
                    f"but got {v}"
                )
            return v

        @field_validator("batch_size")
        def batch_size_check(cls, v):
            if isinstance(v, int) and v <= 0:
                raise ValueError(
                    f"'INFER.batch_size' should be greater than 0 or None, but got {v}"
                )
            return v

    class SolverConfig(BaseModel):
        """
        Schema of global config for pydantic validation.
        """

        # Global settings config
        mode: Literal["train", "eval", "export", "infer"] = "train"
        output_dir: Optional[str] = None
        log_freq: int = 20
        seed: int = 42
        use_vdl: bool = False
        use_tbd: bool = False
        wandb_config: Optional[Mapping] = None
        use_wandb: bool = False
        device: Literal["cpu", "gpu", "xpu"] = "gpu"
        use_amp: bool = False
        amp_level: Literal["O0", "O1", "O2", "OD"] = "O1"
        to_static: bool = False
        prim: bool = False
        log_level: Literal["debug", "info", "warning", "error"] = "info"

        # Training related config
        TRAIN: Optional[TrainConfig] = None

        # Evaluation related config
        EVAL: Optional[EvalConfig] = None

        # Inference related config
        INFER: Optional[InferConfig] = None

        # Fine-grained validator(s) below
        @field_validator("log_freq")
        def log_freq_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'log_freq' should be a non-negtive integer when is type of int"
                    f", but got {v}"
                )
            return v

        @field_validator("seed")
        def seed_check(cls, v):
            if v < 0:
                raise ValueError(f"'seed' should be a non-negtive integer, but got {v}")
            return v

        @field_validator("use_wandb")
        def use_wandb_check(cls, v, info: ValidationInfo):
            if v and not isinstance(info.data["wandb_config"], dict):
                raise ValueError(
                    "'wandb_config' should be a dict when 'use_wandb' is True, "
                    f"but got {info.data['wandb_config'].__class__.__name__}"
                )
            return v

    # Register 'XXXConfig' as default node, so as to be used as default config in *.yaml
    """
    #### xxx.yaml ####
    defaults:
      - ppsci_default             <-- 'ppsci_default' used here
      - TRAIN: train_default      <-- 'train_default' used here
        - TRAIN/ema: ema_default  <-- 'ema_default' used here
        - TRAIN/swa: swa_default  <-- 'swa_default' used here
      - EVAL: eval_default        <-- 'eval_default' used here
      - INFER: infer_default      <-- 'infer_default' used here
      - _self_                    <-- config defined in current yaml

    mode: train
    seed: 42
    ...
    ...
    ##################
    """

    cs = ConfigStore.instance()

    global_default_cfg = SolverConfig().model_dump()
    omegaconf_dict_config = OmegaConf.create(global_default_cfg)
    cs.store(name="ppsci_default", node=omegaconf_dict_config)

    train_default_cfg = TrainConfig().model_dump()
    train_omegaconf_dict_config = OmegaConf.create(train_default_cfg)
    cs.store(group="TRAIN", name="train_default", node=train_omegaconf_dict_config)

    ema_default_cfg = EMAConfig().model_dump()
    ema_omegaconf_dict_config = OmegaConf.create(ema_default_cfg)
    cs.store(group="TRAIN/ema", name="ema_default", node=ema_omegaconf_dict_config)

    swa_default_cfg = SWAConfig().model_dump()
    swa_omegaconf_dict_config = OmegaConf.create(swa_default_cfg)
    cs.store(group="TRAIN/swa", name="swa_default", node=swa_omegaconf_dict_config)

    eval_default_cfg = EvalConfig().model_dump()
    eval_omegaconf_dict_config = OmegaConf.create(eval_default_cfg)
    cs.store(group="EVAL", name="eval_default", node=eval_omegaconf_dict_config)

    infer_default_cfg = InferConfig().model_dump()
    infer_omegaconf_dict_config = OmegaConf.create(infer_default_cfg)
    cs.store(group="INFER", name="infer_default", node=infer_omegaconf_dict_config)

    exclude_keys_default = [
        "mode",
        "output_dir",
        "log_freq",
        "seed",
        "use_vdl",
        "use_tbd",
        "wandb_config",
        "use_wandb",
        "device",
        "use_amp",
        "amp_level",
        "to_static",
        "prim",
        "log_level",
        "TRAIN.save_freq",
        "TRAIN.eval_during_train",
        "TRAIN.start_eval_epoch",
        "TRAIN.eval_freq",
        "TRAIN.checkpoint_path",
        "TRAIN.pretrained_model_path",
        "EVAL.pretrained_model_path",
        "EVAL.eval_with_no_grad",
        "EVAL.compute_metric_by_batch",
        "EVAL.batch_size",
        "INFER.pretrained_model_path",
        "INFER.export_path",
        "INFER.pdmodel_path",
        "INFER.pdiparams_path",
        "INFER.onnx_path",
        "INFER.device",
        "INFER.engine",
        "INFER.precision",
        "INFER.ir_optim",
        "INFER.min_subgraph_size",
        "INFER.gpu_mem",
        "INFER.gpu_id",
        "INFER.max_batch_size",
        "INFER.num_cpu_threads",
        "INFER.batch_size",
    ]
    cs.store(
        group="hydra/job/config/override_dirname/exclude_keys",
        name="exclude_keys_default",
        node=exclude_keys_default,
    )

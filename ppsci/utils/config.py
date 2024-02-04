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

import argparse
import copy
import importlib.util
import os
from typing import Mapping
from typing import Optional

import yaml
from paddle import static
from typing_extensions import Literal

from ppsci.utils import logger
from ppsci.utils import misc

__all__ = ["get_config", "replace_shape_with_inputspec_", "AttrDict"]

if importlib.util.find_spec("pydantic") is not None:
    from pydantic import BaseModel
    from pydantic import field_validator
    from pydantic_core.core_schema import FieldValidationInfo

    __all__.append("SolverConfig")

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

        # Fine-grained validator(s) below
        @field_validator("epochs")
        def epochs_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'epochs' should be a positive integer when is type of int, "
                    f"but got {v}"
                )
            return v

        @field_validator("iters_per_epoch")
        def iters_per_epoch_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'iters_per_epoch' should be a positive integer when is type of int"
                    f", but got {v}"
                )
            return v

        @field_validator("update_freq")
        def update_freq_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'update_freq' should be a positive integer when is type of int"
                    f", but got {v}"
                )
            return v

        @field_validator("save_freq")
        def save_freq_check(cls, v):
            if v < 0:
                raise ValueError(
                    "'save_freq' should be a non-negtive integer when is type of int"
                    f", but got {v}"
                )
            return v

        @field_validator("start_eval_epoch")
        def start_eval_epoch_check(cls, v, info: FieldValidationInfo):
            if info.data["eval_during_train"]:
                if v <= 0:
                    raise ValueError(
                        f"'start_eval_epoch' should be a positive integer when "
                        f"'eval_during_train' is True, but got {v}"
                    )
            return v

        @field_validator("eval_freq")
        def eval_freq_check(cls, v, info: FieldValidationInfo):
            if info.data["eval_during_train"]:
                if v <= 0:
                    raise ValueError(
                        f"'eval_freq' should be a positive integer when "
                        f"'eval_during_train' is True, but got {v}"
                    )
            return v

    class EvalConfig(BaseModel):
        """
        Schema of evaluation config for pydantic validation.
        """

        pretrained_model_path: Optional[str] = None
        eval_with_no_grad: bool = False
        compute_metric_by_batch: bool = False

    class InferConfig(BaseModel):
        """
        Schema of inference config for pydantic validation.
        """

        pretrained_model_path: Optional[str] = None
        export_path: str
        pdmodel_path: Optional[str] = None
        pdpiparams_path: Optional[str] = None
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
        batch_size: int = 256

        # Fine-grained validator(s) below
        @field_validator("engine")
        def engine_check(cls, v, info: FieldValidationInfo):
            if v == "tensorrt" and info.data["device"] != "gpu":
                raise ValueError(
                    "'device' should be 'gpu' when 'engine' is 'tensorrt', "
                    f"but got '{info.data['device']}'"
                )
            if v == "mkldnn" and info.data["device"] != "cpu":
                raise ValueError(
                    "'device' should be 'cpu' when 'engine' is 'mkldnn', "
                    f"but got '{info.data['device']}'"
                )

            return v

        @field_validator("min_subgraph_size")
        def min_subgraph_size_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'min_subgraph_size' should be greater than 0, " f"but got {v}"
                )
            return v

        @field_validator("gpu_mem")
        def gpu_mem_check(cls, v):
            if v <= 0:
                raise ValueError("'gpu_mem' should be greater than 0, " f"but got {v}")
            return v

        @field_validator("gpu_id")
        def gpu_id_check(cls, v):
            if v < 0:
                raise ValueError(
                    "'gpu_id' should be greater than or equal to 0, " f"but got {v}"
                )
            return v

        @field_validator("max_batch_size")
        def max_batch_size_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'max_batch_size' should be greater than 0, " f"but got {v}"
                )
            return v

        @field_validator("num_cpu_threads")
        def num_cpu_threads_check(cls, v):
            if v < 0:
                raise ValueError(
                    "'num_cpu_threads' should be greater than or equal to 0, "
                    f"but got {v}"
                )
            return v

        @field_validator("batch_size")
        def batch_size_check(cls, v):
            if v <= 0:
                raise ValueError(
                    "'batch_size' should be greater than 0, " f"but got {v}"
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
        use_wandb: bool = False
        wandb_config: Optional[Mapping] = None
        device: Literal["cpu", "gpu", "xpu"] = "gpu"
        use_amp: bool = False
        amp_level: Literal["O0", "O1", "O2", "OD"] = "O1"
        to_static: bool = False
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
        def use_wandb_check(cls, v, info: FieldValidationInfo):
            if not isinstance(info.data["wandb_config"], dict):
                raise ValueError(
                    "'wandb_config' should be a dict when 'use_wandb' is True, "
                    f"but got {misc.typename(info.data['wandb_config'])}"
                )
            return v


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return AttrDict(copy.deepcopy(dict(self)))


def create_attr_dict(yaml_config):
    from ast import literal_eval

    for key, value in yaml_config.items():
        if isinstance(value, dict):
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, "r") as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting according by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in d.items():
        if isinstance(v, dict):
            logger.info(f"{delimiter * ' '}{k} : ")
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info(f"{delimiter * ' '}{k} : ")
            for value in v:
                print_dict(value, delimiter + 2)
        else:
            logger.info(f"{delimiter * ' '}{k} : {v}")

        if k[0].isupper() and delimiter == 0:
            logger.info(placeholder)


def print_config(config):
    """
    Visualize configs
    Arguments:
        config: configs
    """
    logger.advertise()
    print_dict(config)


def override(dl, ks, v):
    """
    Recursively replace dict of list
    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """

    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    if not isinstance(dl, (list, dict)):
        raise ValueError(f"{dl} should be a list or a dict")
    if len(ks) <= 0:
        raise ValueError("length of keys should be larger than 0")

    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            if k >= len(dl):
                raise ValueError(f"index({k}) out of range({dl})")
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, (f"{ks[0]} is not exist in {dl}")
            if ks[0] not in dl:
                print(f"A new field ({ks[0]}) detected!")
            dl[ks[0]] = str2num(v)
        else:
            if ks[0] not in dl.keys():
                dl[ks[0]] = {}
                print(f"A new Series field ({ks[0]}) detected!")
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                "topk=2",
                "VALID.transforms.1.ResizeImage.resize_short=300"
            ]
    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), f"option({opt}) should be a str"
            assert (
                "=" in opt
            ), f"option({opt}) should contain a = to distinguish between key and value"
            pair = opt.split("=")
            assert len(pair) == 2, "there can be only a = in the option"
            key, value = pair
            keys = key.split(".")
            override(config, keys, value)
    return config


def get_config(fname, overrides=None, show=False):
    """
    Read config from file
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"config file({fname}) is not exist")
    config = parse_config(fname)
    override_config(config, overrides)
    if show:
        print_config(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser("paddlescience running script")
    parser.add_argument("-e", "--epochs", type=int, help="training epochs")
    parser.add_argument("-o", "--output_dir", type=str, help="output directory")
    parser.add_argument(
        "--to_static",
        action="store_true",
        help="whether enable to_static for forward computation",
    )

    args = parser.parse_args()
    return args


def _is_num_seq(seq):
    # whether seq is all int number(it is a shape)
    return isinstance(seq, (list, tuple)) and all(isinstance(x, int) for x in seq)


def replace_shape_with_inputspec_(node: AttrDict):
    if _is_num_seq(node):
        return True

    if isinstance(node, dict):
        for key in node:
            if replace_shape_with_inputspec_(node[key]):
                node[key] = static.InputSpec(node[key])
    elif isinstance(node, list):
        for i in range(len(node)):
            if replace_shape_with_inputspec_(node[i]):
                node[i] = static.InputSpec(node[i])

    return False

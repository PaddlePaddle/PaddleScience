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

import argparse
import copy
import os

import yaml
from paddle import static

from ppsci.utils import logger

__all__ = ["get_config", "replace_shape_with_inputspec_", "AttrDict"]


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
    indenting acrrording by the relationship of keys.
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
        raise ValueError("lenght of keys should be larger than 0")

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
            if not ks[0] in dl:
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

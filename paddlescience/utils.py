# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import yaml
import argparse
from .config import enable_visualdl, enable_static, enable_prim

__all__ = ['parse_args']


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
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


def parse_config(cfg_file, config_index):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = yaml.load(fopen, Loader=yaml.SafeLoader)
    create_attr_dict(yaml_config[config_index])
    return yaml_config[config_index]


def get_config(fname, overrides=None, config_index=0):
    """
    Read config from file
    """
    if fname is None:
        return None
    assert os.path.exists(fname), (
        'config file({}) is not exist'.format(fname))
    config = parse_config(fname, config_index)

    return config


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleScience')
    parser.add_argument(
        '-c', '--config-file', metavar="FILE", help='config file path')

    parser.add_argument(
        '-i',
        '--config-index',
        type=int,
        default=0,
        help='run validation every interval')

    args = parser.parse_args()

    # Get config
    cfg = get_config(args.config_file, args.opt, args.config_index)

    # Enable related flags
    if cfg['visualdl_enabled'] == True:
        enable_visualdl()
    if cfg['static_enabled'] == True:
        enable_static()
    if cfg['prim_enabled'] == True:
        enable_prim()

    return cfg

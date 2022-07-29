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

import argparse
from .config import get_config
from ..config import enable_visualdl, enable_static, enable_prim


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

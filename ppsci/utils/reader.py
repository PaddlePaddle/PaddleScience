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

import collections
import csv

import numpy as np
import scipy.io as sio

from ppsci.utils import logger


def load_csv_file(file_path, keys, alias_dict=None, encoding="utf-8"):
    if alias_dict is None:
        alias_dict = {}

    try:
        # read all data from csv file
        with open(file_path, "r", encoding=encoding) as csv_file:
            reader = csv.DictReader(csv_file)
            raw_data = collections.defaultdict(list)
            for _, line_dict in enumerate(reader):
                for key, value in line_dict.items():
                    raw_data[key].append(value)

        # convert to numpy array
        data_dict = {}
        for key in keys:
            fetch_key = alias_dict[key] if key in alias_dict else key
            if fetch_key not in raw_data:
                raise KeyError(f"fetch_key({fetch_key}) do not exist in raw_data.")
            data_dict[key] = np.asarray(raw_data[fetch_key], "float32").reshape([-1, 1])

        return data_dict
    except Exception as e:
        logger.error(f"{repr(e)}, {file_path} isn't a valid csv file.")
        raise


def load_mat_file(file_path, keys, alias_dict=None):
    if alias_dict is None:
        alias_dict = {}

    try:
        # read all data from mat file
        raw_data = sio.loadmat(file_path)

        # convert to numpy array
        data_dict = {}
        for key in keys:
            fetch_key = alias_dict[key] if key in alias_dict else key
            if fetch_key not in raw_data:
                raise KeyError(f"fetch_key({fetch_key}) do not exist in raw_data.")
            data_dict[key] = np.asarray(raw_data[fetch_key], "float32").reshape([-1, 1])

        return data_dict

    except Exception as e:
        logger.error(f"{repr(e)}, {file_path} isn't a valid mat file.")
        raise

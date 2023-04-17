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

    # check if all target keys exist in keys
    for target_key, original_key in alias_dict.items():
        if target_key not in keys:
            raise ValueError(
                f"target_key({target_key}) in alias_dict "
                f"is not found in keys({keys}) for mapping"
                f" {original_key}->{target_key}."
            )
    try:
        # read all data from csv file
        with open(file_path, "r", encoding=encoding) as csv_file:
            reader = csv.DictReader(csv_file)
            raw_data = collections.defaultdict(list)
            for line_idx, line_dict in enumerate(reader):
                for key, value in line_dict.items():
                    raw_data[key].append(value)
                # check if all keys are available at first line
                if line_idx == 0:
                    for require_key in keys:
                        if require_key not in line_dict:
                            raise KeyError(
                                f"key({require_key}) "
                                f"not found in csv file({file_path})"
                            )

            # check if all original keys exist in raw_data
            for original_key in alias_dict.values():
                if original_key not in raw_data:
                    raise ValueError(
                        f"original_key({original_key}) not exist in raw_data."
                    )

            data_dict = {}
            for key in keys:
                if key in alias_dict:
                    data_dict[key] = np.asarray(
                        raw_data[alias_dict[key]], "float32"
                    ).reshape([-1, 1])
                else:
                    data_dict[key] = np.asarray(raw_data[key], "float32").reshape(
                        [-1, 1]
                    )

        return data_dict
    except Exception as e:
        logger.error(f"{repr(e)}, {file_path} isn't a valid csv file.")
        raise


def load_mat_file(file_path, keys, alias_dict=None):
    if alias_dict is None:
        alias_dict = {}

    # check if all target keys exist in keys
    for target_key, original_key in alias_dict.items():
        if not target_key in keys:
            raise ValueError(
                f"target_key({target_key}) in alias_dict "
                f"is not found in keys({keys}) for mapping"
                f" {original_key}->{target_key}."
            )
    try:
        raw_data = sio.loadmat(file_path)
        # check if all original keys exist in raw_data
        for original_key in alias_dict.values():
            if original_key not in raw_data:
                raise ValueError(f"original_key({original_key}) not exist in raw_data.")
        # convert to numpy array
        data_dict = {}
        for key in keys:
            if key in alias_dict:
                data_dict[key] = np.asarray(
                    raw_data[alias_dict[key]], "float32"
                ).reshape([-1, 1])
            else:
                data_dict[key] = np.asarray(raw_data[key], "float32").reshape([-1, 1])

        return data_dict

    except Exception as e:
        logger.error(f"{repr(e)}, {file_path} isn't a valid mat file.")
        raise

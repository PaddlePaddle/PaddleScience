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
import sys
from typing import Dict
from typing import Optional
from typing import Tuple

import meshio
import numpy as np
import paddle
import scipy.io as sio

from ppsci.utils import logger

__all__ = [
    "load_csv_file",
    "load_mat_file",
    "load_vtk_file",
    "load_vtk_with_time_file",
]


def load_csv_file(
    file_path: str,
    keys: Tuple[str, ...],
    alias_dict: Optional[Dict[str, str]] = None,
    delimeter: str = ",",
    encoding: str = "utf-8",
) -> Dict[str, np.ndarray]:
    """Load *.csv file and fetch data as given keys.

    Args:
        file_path (str): CSV file path.
        keys (Tuple[str, ...]): Required fetching keys.
        alias_dict (Optional[Dict[str, str]]): Alias for keys,
            i.e. {inner_key: outer_key}. Defaults to None.
        encoding (str, optional): Encoding code when open file. Defaults to "utf-8".

    Returns:
        Dict[str, np.ndarray]: Loaded data in dict.
    """
    if alias_dict is None:
        alias_dict = {}

    try:
        # read all data from csv file
        with open(file_path, "r", encoding=encoding) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=delimeter)
            raw_data = collections.defaultdict(list)
            for _, line_dict in enumerate(reader):
                for key, value in line_dict.items():
                    raw_data[key].append(value)
    except FileNotFoundError:
        logger.error(f"{file_path} isn't a valid csv file.")
        sys.exit()

    # convert to numpy array
    data_dict = {}
    for key in keys:
        fetch_key = alias_dict[key] if key in alias_dict else key
        if fetch_key not in raw_data:
            raise KeyError(f"fetch_key({fetch_key}) do not exist in raw_data.")
        data_dict[key] = np.asarray(
            raw_data[fetch_key], paddle.get_default_dtype()
        ).reshape([-1, 1])

    return data_dict


def load_mat_file(
    file_path: str, keys: Tuple[str, ...], alias_dict: Optional[Dict[str, str]] = None
) -> Dict[str, np.ndarray]:
    """Load *.mat file and fetch data as given keys.

    Args:
        file_path (str): Mat file path.
        keys (Tuple[str, ...]): Required fetching keys.
        alias_dict (Optional[Dict[str, str]]): Alias for keys,
            i.e. {original_key: original_key}. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Loaded data in dict.
    """

    if alias_dict is None:
        alias_dict = {}

    try:
        # read all data from mat file
        raw_data = sio.loadmat(file_path)
    except FileNotFoundError:
        logger.error(f"{file_path} isn't a valid mat file.")
        raise

    # convert to numpy array
    data_dict = {}
    for key in keys:
        fetch_key = alias_dict[key] if key in alias_dict else key
        if fetch_key not in raw_data:
            raise KeyError(f"fetch_key({fetch_key}) do not exist in raw_data.")
        data_dict[key] = np.asarray(
            raw_data[fetch_key], paddle.get_default_dtype()
        ).reshape([-1, 1])

    return data_dict


def load_vtk_file(
    filename_without_timeid: str,
    time_step: float,
    time_index: Tuple[int, ...],
    input_keys: Tuple[str, ...],
    label_keys: Optional[Tuple[str, ...]],
) -> Dict[str, np.ndarray]:
    """Load coordinates and attached label from the *.vtu file.

    Args:
        filename_without_timeid (str): File name without time id.
        time_step (float): Physical time step.
        time_index (Tuple[int, ...]): Physical time indexes.
        input_keys (Tuple[str, ...]): Input coordinates name keys.
        label_keys (Optional[Tuple[str, ...]]): Input label name keys.

    Returns:
        Dict[str, np.ndarray]: Input coordinates dict, label coordinates dict
    """
    input_dict = {var: [] for var in input_keys}
    label_dict = {var: [] for var in label_keys}
    for index in time_index:
        file = filename_without_timeid + f"{index}.vtu"
        mesh = meshio.read(file)
        n = mesh.points.shape[0]
        i = 0
        for key in input_dict:
            if key == "t":
                input_dict[key].append(np.full((n, 1), index * time_step, "float32"))
            else:
                input_dict[key].append(
                    mesh.points[:, i].reshape(n, 1).astype("float32")
                )
                i += 1
        for i, key in enumerate(label_dict):
            label_dict[key].append(np.array(mesh.point_data[key], "float32"))
    for key in input_dict:
        input_dict[key] = np.concatenate(input_dict[key])
    for key in label_dict:
        label_dict[key] = np.concatenate(label_dict[key])

    return input_dict, label_dict


def load_vtk_with_time_file(file: str) -> Dict[str, np.ndarray]:
    """Temporary interface for points cloud, will be banished sooner.

    Args:
        file (str): input file name.

    Returns:
        Dict[str, np.ndarray]: Input coordinates dict.
    """
    mesh = meshio.read(file)
    n = mesh.points.shape[0]
    t = np.array(mesh.point_data["time"])
    x = mesh.points[:, 0].reshape(n, 1)
    y = mesh.points[:, 1].reshape(n, 1)
    z = mesh.points[:, 2].reshape(n, 1)
    input_dict = {"t": t, "x": x, "y": y, "z": z}
    return input_dict

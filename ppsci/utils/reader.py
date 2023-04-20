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

import meshio
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


def load_vtk_file(
    filename_without_timeid: str,
    time_step,
    time_index,
    read_input: bool = True,
    read_label: bool = True,
    dim=3,
):
    for i, t in enumerate(time_index):
        file = filename_without_timeid + f"{t}.vtu"
        mesh = meshio.read(file)
        if i == 0:
            n = mesh.points.shape[0]
            input_dict = {
                var: np.zeros((len(time_index) * n, 1)).astype(np.float32)
                for var in ["t", "x", "y", "z"]
            }
            label_dict = {
                var: np.zeros((len(time_index) * n, 1)).astype(np.float32)
                for var in ["u", "v", "w", "p"]
            }
        if read_input == True:
            input_dict["t"][i * n : (i + 1) * n] = np.full((n, 1), int(t * time_step))
            input_dict["x"][i * n : (i + 1) * n] = mesh.points[:, 0].reshape(n, 1)
            input_dict["y"][i * n : (i + 1) * n] = mesh.points[:, 1].reshape(n, 1)
            if dim == 3:
                input_dict["z"][i * n : (i + 1) * n] = mesh.points[:, 2].reshape(n, 1)
        if read_label == True:
            label_dict["u"][i * n : (i + 1) * n] = np.array(mesh.point_data["1"])
            label_dict["v"][i * n : (i + 1) * n] = np.array(mesh.point_data["2"])
            if dim == 3:
                label_dict["w"][i * n : (i + 1) * n] = np.array(mesh.point_data["3"])
            label_dict["p"][i * n : (i + 1) * n] = np.array(mesh.point_data["4"])
    return input_dict, label_dict


def load_vtk_withtime_file(file: str):
    mesh = meshio.read(file)
    n = mesh.points.shape[0]
    t = np.array(mesh.point_data["time"])
    x = mesh.points[:, 0].reshape(n, 1)
    y = mesh.points[:, 1].reshape(n, 1)
    z = mesh.points[:, 2].reshape(n, 1)
    txyz = np.concatenate((t, x, y, z), axis=1).astype(np.float32).reshape(n, 4, 1)
    input_dict = {"t": t, "x": x, "y": y, "z": z}
    return input_dict

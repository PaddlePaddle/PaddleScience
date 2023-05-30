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

import glob
from typing import Dict
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import paddle
from paddle import io
from paddle import vision


class ERA5Dataset(io.Dataset):
    """Class for ERA5 dataset.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        precip_file_path (Optional[str]): Precipitation data set path. Defaults to None.
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        vars_channel (Optional[Tuple[int, ...]]): The variable channel index in ERA5 dataset. Defaults to None.
        num_label_timestamps (int, optional): Number of timestamp of label. Defaults to 1.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.
        training (bool, optional): Whether in train mode. Defaults to True.
        stride (int, optional): Stride of sampling data. Defaults to 1.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.ERA5Dataset(
        ...     "file_path": "/path/to/ERA5Dataset",
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        precip_file_path: Optional[str] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        vars_channel: Optional[Tuple[int, ...]] = None,
        num_label_timestamps: int = 1,
        transforms: Optional[vision.Compose] = None,
        training: bool = True,
        stride: int = 1,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.precip_file_path = precip_file_path

        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.vars_channel = list(range(20)) if vars_channel is None else vars_channel
        self.num_label_timestamps = num_label_timestamps
        self.transforms = transforms
        self.training = training
        self.stride = stride

        self.files = self.read_data(file_path)
        self.n_years = len(self.files)
        self.num_samples_per_year = self.files[0].shape[0]
        self.num_samples = self.n_years * self.num_samples_per_year
        if self.precip_file_path is not None:
            self.precip_files = self.read_data(precip_file_path, "tp")

    def read_data(self, path: str, var="fields"):
        paths = [path] if path.endswith(".h5") else glob.glob(path + "/*.h5")
        paths.sort()
        files = []
        for path_ in paths:
            _file = h5py.File(path_, "r")
            files.append(_file[var])
        return files

    def __len__(self):
        return self.num_samples // self.stride

    def __getitem__(self, global_idx):
        global_idx *= self.stride
        year_idx = global_idx // self.num_samples_per_year
        local_idx = global_idx % self.num_samples_per_year
        step = 0 if local_idx >= self.num_samples_per_year - 1 else 1

        if self.num_label_timestamps > 1:
            if local_idx >= self.num_samples_per_year - self.num_label_timestamps:
                local_idx = self.num_samples_per_year - self.num_label_timestamps - 1

        input_file = self.files[year_idx]
        label_file = (
            self.precip_files[year_idx]
            if self.precip_file_path is not None
            else input_file
        )
        if self.precip_file_path is not None and year_idx == 0 and self.training:
            # first year has 2 missing samples in precip (they are first two time points)
            lim = self.num_samples_per_year - 2
            local_idx = local_idx % lim
            step = 0 if local_idx >= lim - 1 else 1
            input_idx = local_idx + 2
            label_idx = local_idx + step
        else:
            input_idx, label_idx = local_idx, local_idx + step

        input_item = {self.input_keys[0]: input_file[input_idx, self.vars_channel]}

        label_item = {}
        for i in range(self.num_label_timestamps):
            if self.precip_file_path is not None:
                label_item[self.label_keys[i]] = np.expand_dims(
                    label_file[label_idx + i], 0
                )
            else:
                label_item[self.label_keys[i]] = label_file[
                    label_idx + i, self.vars_channel
                ]

        weight_shape = [1] * len(next(iter(label_item.values)).shape)
        weight_item = {
            key: np.full(weight_shape, value, paddle.get_default_dtype())
            for key, value in self.weight_dict.items()
        }

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                (input_item, label_item, weight_item)
            )

        return input_item, label_item, weight_item


class ERA5SampledDataset(io.Dataset):
    """Class for ERA5 sampled dataset.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.
    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.ERA5SampledDataset(
        ...     "file_path": "/path/to/ERA5SampledDataset",
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys

        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.transforms = transforms

        self.files = self.read_data(file_path)
        self.num_samples = len(self.files)

    def read_data(self, path: str):
        paths = glob.glob(path + "/*.h5")
        paths.sort()
        files = []
        for _path in paths:
            _file = h5py.File(_path, "r")
            files.append(_file)
        return files

    def __len__(self):
        return self.num_samples

    def __getitem__(self, global_idx):
        _file = self.files[global_idx]

        input_item = {}
        for key in _file["input_dict"]:
            input_item[key] = np.asarray(
                _file["input_dict"][key], paddle.get_default_dtype()
            )

        label_item = {}
        for key in _file["label_dict"]:
            label_item[key] = np.asarray(
                _file["label_dict"][key], paddle.get_default_dtype()
            )

        weight_shape = [1] * len(next(iter(label_item.values)).shape)
        weight_item = {
            key: np.full(weight_shape, value, paddle.get_default_dtype())
            for key, value in self.weight_dict.items()
        }

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                (input_item, label_item, weight_item)
            )

        return input_item, label_item, weight_item

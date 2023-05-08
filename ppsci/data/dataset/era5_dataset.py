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
from paddle import io
from paddle import vision


class ERA5Dataset(io.Dataset):
    """Class for ERA5 dataset.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        precip_file_path (str, optional): Precipitation data set path. Defaults to None.
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        vars_channel (Tuple[int, ...], optional): The variable channel index in ERA5 dataset. Defaults to None.
        label_timestamp (int, optional): Number of label timestamp. Defaults to 1.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.
        train (bool, optional): Whether in train mode. Defaults to True.

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
        precip_file_path: str = None,
        weight_dict: Optional[Dict[str, float]] = None,
        vars_channel: Tuple[int, ...] = None,
        label_timestamp: int = 1,
        transforms: Optional[vision.Compose] = None,
        train: bool = True,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.precip_file_path = precip_file_path
        self.precip = False if precip_file_path is None else True

        self.weight_dict = weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        if vars_channel is None:
            self.vars_channel = [i for i in range(20)]
        else:
            self.vars_channel = vars_channel
        self.label_timestamp = label_timestamp
        self.transforms = transforms
        self.train = train

        self.files = self.read_data(file_path)
        self.n_years = len(self.files)
        self.n_samples_per_year = self.files[0].shape[0]
        self.n_samples_total = self.n_years * self.n_samples_per_year
        if self.precip is True:
            self.precip_files = self.read_data(precip_file_path, "tp")

    def read_data(self, path: str, var="fields"):
        paths = glob.glob(path + "/*.h5")
        paths.sort()
        files = []
        for path in paths:
            _file = h5py.File(path, "r")
            files.append(_file[var])
        return files

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = global_idx // self.n_samples_per_year
        local_idx = global_idx % self.n_samples_per_year
        step = 0 if local_idx >= self.n_samples_per_year - 1 else 1

        if self.label_timestamp > 1:
            if local_idx >= self.n_samples_per_year - self.label_timestamp:
                local_idx = self.n_samples_per_year - self.label_timestamp - 1

        input_file = self.files[year_idx]
        label_file = self.precip_files[year_idx] if self.precip else input_file
        if self.precip is True and year_idx == 0 and self.train:
            # first year has 2 missing samples in precip (they are first two time points)
            lim = self.n_samples_per_year - 2
            local_idx = local_idx % lim
            step = 0 if local_idx >= lim - 1 else 1
            input_idx = local_idx + 2
            label_idx = local_idx + step
        else:
            input_idx, label_idx = local_idx, local_idx + step

        input_item = {self.input_keys[0]: input_file[input_idx, self.vars_channel]}
        label_item = {}
        for i in range(self.label_timestamp):
            if self.precip is True:
                label_item[self.label_keys[i]] = np.expand_dims(
                    label_file[label_idx + i], 0
                )
            else:
                label_item[self.label_keys[i]] = label_file[
                    label_idx + i, self.vars_channel
                ]

        if self.weight_dict is not None:
            weight_shape = [1] * len(next(iter(label_item.values)).shape)
            weight_item = {
                key: np.full(weight_shape, value)
                for key, value in self.weight_dict.items()
            }
        else:
            weight_item = None

        data = (input_item, label_item, weight_item)
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def getitem(self, global_idx):
        return self.__getitem__(global_idx)


class ERA5SampledDataset(io.Dataset):
    """Class for ERA5 sampled dataset.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise

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
        transforms=None,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys

        self.weight_dict = weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.transforms = transforms

        self.files = self.read_data(file_path)
        self.n_samples_total = len(self.files)

    def read_data(self, path: str):
        paths = glob.glob(path + "/*.h5")
        paths.sort()
        files = []
        for path in paths:
            _file = h5py.File(path, "r")
            files.append(_file)
        return files

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        _file = self.files[global_idx]

        input_item = {}
        for key in _file["input_dict"]:
            input_item[key] = np.asarray(_file["input_dict"][key])
        label_item = {}
        for key in _file["label_dict"]:
            label_item[key] = np.asarray(_file["label_dict"][key])

        if self.weight_dict is not None:
            weight_shape = [1] * len(next(iter(label_item.values)).shape)
            weight_item = {
                key: np.full(weight_shape, value)
                for key, value in self.weight_dict.items()
            }
        else:
            weight_item = None

        data = (input_item, label_item, weight_item)
        if self.transforms is not None:
            data = self.transforms(data)

        return data

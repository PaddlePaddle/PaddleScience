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

import os
from typing import Dict
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import paddle
from paddle import io
from paddle import vision


class HRRRDataset(io.Dataset):
    """Class for HRRR dataset.

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
        weight_dict: Optional[Dict[str, float]] = None,
        num_label_timestamps: int = 1,
        vars_channel: Optional[Tuple[int, ...]] = None,
        transforms: Optional[vision.Compose] = None,
        training: bool = True,
        stride: int = 1,
        lead_time: int = 1,
        extra_file_path: Optional[str] = None,
        extra_vars_channel: Optional[Tuple[int, ...]] = None,
        merge_label: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys

        self.weight_dict = {key: 1.0 for key in self.label_keys}
        if weight_dict is not None:
            self.weight_dict.update(weight_dict)

        self.vars_channel = list(range(69)) if vars_channel is None else vars_channel
        self.num_label_timestamps = num_label_timestamps
        self.transforms = transforms
        self.training = training
        self.stride = stride
        self.lead_time = lead_time
        self.extra_file_path = extra_file_path
        self.extra_vars_channel = extra_vars_channel
        self.merge_label = merge_label

        self.files = self.read_data(file_path, extra_file_path)
        self.num_days = len(self.files)
        self.num_samples_per_day = self.files[0][0].shape[0]
        self.num_samples = self.num_days * self.num_samples_per_day

    def read_data(self, path: str, extra_path: str, var="fields"):
        if path.endswith(".h5"):
            paths = [path]
        else:
            paths = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    paths.append(os.path.join(root, file))
        paths.sort()
        files = []
        for path_ in paths:
            _file = h5py.File(path_, "r")
            if extra_path is not None:
                _extra_file = h5py.File(os.path.join(extra_path, path_[-13:]), "r")
                files.append([_file[var], path_[-13:-3], _extra_file[var]])
            else:
                files.append([_file[var], path_[-13:-3]])

        return files

    def __len__(self):
        return self.num_samples // self.stride

    def __getitem__(self, global_idx):

        global_idx *= self.stride

        if global_idx >= self.num_samples - self.num_label_timestamps - self.lead_time:
            return self.__getitem__(np.random.randint(self.__len__()))

        input_day_idx = global_idx // self.num_samples_per_day
        input_hour_idx = global_idx % self.num_samples_per_day

        input_file = self.files[input_day_idx][0]
        # check fake data
        if len(input_file.shape) == 1:
            print("Warning: fake data detected, please check your data")
            return self.__getitem__(np.random.randint(self.__len__()))
        input_item = {self.input_keys[0]: input_file[input_hour_idx, self.vars_channel]}
        if self.extra_file_path is not None:
            extra_input = self.files[input_day_idx][2][
                input_hour_idx, self.extra_vars_channel
            ]
            input_item[self.input_keys[0]] = np.concatenate(
                [input_item[self.input_keys[0]], extra_input]
            )

        # label_item = {self.label_keys[0]: label_file[label_hour_idx, self.vars_channel]}
        input_time_list = []
        input_time = self.files[input_day_idx][1] + "/" + str(input_hour_idx)
        input_time_list.append(input_time)

        label_item = {}
        label_time = {}

        for i in range(self.num_label_timestamps):
            label_day_idx = (
                global_idx + self.lead_time + i
            ) // self.num_samples_per_day
            label_hour_idx = (
                global_idx + self.lead_time + i
            ) % self.num_samples_per_day
            label_file = self.files[label_day_idx][0]
            if len(label_file.shape) == 1:
                print("Warning: fake data detected, please check your data")
                return self.__getitem__(np.random.randint(self.__len__()))
            label_item[self.label_keys[i]] = label_file[
                label_hour_idx, self.vars_channel
            ]
            if self.extra_file_path is not None:
                extra_label = self.files[label_day_idx][2][
                    label_hour_idx, self.extra_vars_channel
                ]
                label_item[self.label_keys[i]] = np.concatenate(
                    [label_item[self.label_keys[i]], extra_label]
                )

            label_time[self.label_keys[i]] = (
                self.files[label_day_idx][1] + "/" + str(label_hour_idx)
            )
            input_time = self.files[label_day_idx][1] + "/" + str(label_hour_idx)
            input_time_list.append(input_time)
        # merge label
        if self.merge_label:
            for i in range(self.num_label_timestamps):
                input_item[f"{self.input_keys[0]}_{i}_merge"] = label_item[
                    self.label_keys[i]
                ]
        # import remote_pdb as pdb;pdb.set_trace()
        weight_shape = [1] * len(next(iter(label_item.values())).shape)
        weight_item = {
            key: np.full(weight_shape, value, paddle.get_default_dtype())
            for key, value in self.weight_dict.items()
        }

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                (input_item, label_item, weight_item)
            )

        return input_item, label_item, weight_item, input_time_list


class HRRRDatasetMultiInput(HRRRDataset):
    """Class for HRRR dataset.

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
        weight_dict: Optional[Dict[str, float]] = None,
        num_input_timestamps: int = 1,
        num_label_timestamps: int = 1,
        vars_channel: Optional[Tuple[int, ...]] = None,
        transforms: Optional[vision.Compose] = None,
        training: bool = True,
        stride: int = 1,
    ):
        super().__init__(
            file_path=file_path,
            input_keys=input_keys,
            label_keys=label_keys,
            weight_dict=weight_dict,
            num_label_timestamps=num_label_timestamps,
            vars_channel=vars_channel,
            transforms=transforms,
            training=training,
            stride=stride,
        )
        self.num_input_timestamps = num_input_timestamps

    def __len__(self):
        return (self.num_samples - self.num_input_timestamps) // self.stride

    def __getitem__(self, global_idx):

        global_idx = global_idx * self.stride + self.num_input_timestamps

        if (
            global_idx < (self.num_input_timestamps - 1)
            or global_idx >= self.num_samples - self.num_label_timestamps
        ):
            return self.__getitem__(np.random.randint(self.__len__()))

        input_item = {}
        for i in range(self.num_input_timestamps):

            input_day_idx = (global_idx - i) // self.num_samples_per_day
            input_hour_idx = (global_idx - i) % self.num_samples_per_day

            input_file = self.files[input_day_idx]
            # check fake data
            if len(input_file.shape) == 1:
                print("Warning: fake data detected, please check your data")
                return self.__getitem__(np.random.randint(self.__len__()))
            input_item[self.input_keys[i]] = input_file[
                input_hour_idx, self.vars_channel
            ]
        # input_item = {self.input_keys[0]: input_file[input_hour_idx, self.vars_channel]}
        # label_item = {self.label_keys[0]: label_file[label_hour_idx, self.vars_channel]}

        label_item = {}
        for i in range(self.num_label_timestamps):
            label_day_idx = (global_idx + 1 + i) // self.num_samples_per_day
            label_hour_idx = (global_idx + 1 + i) % self.num_samples_per_day
            label_file = self.files[label_day_idx]
            if len(label_file.shape) == 1:
                print("Warning: fake data detected, please check your data")
                return self.__getitem__(np.random.randint(self.__len__()))
            label_item[self.label_keys[i]] = label_file[
                label_hour_idx, self.vars_channel
            ]
        weight_shape = [1] * len(next(iter(label_item.values())).shape)
        weight_item = {
            key: np.full(weight_shape, value, paddle.get_default_dtype())
            for key, value in self.weight_dict.items()
        }

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                (input_item, label_item, weight_item)
            )

        return input_item, label_item, weight_item

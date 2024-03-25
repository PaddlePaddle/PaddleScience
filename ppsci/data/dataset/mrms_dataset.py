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

from __future__ import annotations

import glob
import os.path as osp
from datetime import datetime
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import paddle
from paddle import io
from paddle import vision


class MRMSDataset(io.Dataset):
    """Class for MRMS dataset. MRMS day's data is stored in a .h5 file. Each file includes keys "date"/"time_interval"/"dataset".

    Args:
        file_path (str): Dataset path.
        input_keys (Tuple[str, ...]): Input keys, usually there is only one, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, usually there is only one, such as ("output",).
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        date_period (Tuple[str,...], optional): Dates of data. Scale is [start_date, end_date] with format "%Y%m%d". Defaults to ("20230101","20230101").
        num_input_timestamps (int, optional): Number of timestamp of input. Defaults to 1.
        num_label_timestamps (int, optional): Number of timestamp of label. Defaults to 1.
        stride (int, optional): Stride of sampling data. Defaults to 1.
        transforms (Optional[vision.Compose]): Composed transform functor(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.MRMSDataset(
        ...     "file_path": "/path/to/MRMSDataset",
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "date_period": ("20230101","20230131"),
        ...     "num_input_timestamps": 9,
        ...     "num_label_timestamps": 20,
        ...     "transforms": transform,
        ...     "stride": 1,
        ... )  # doctest: +SKIP
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        date_period: Tuple[str, ...] = ("20230101", "20230101"),
        num_input_timestamps: int = 1,
        num_label_timestamps: int = 1,
        stride: int = 1,
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

        self.date_list = self._get_date_strs(date_period)
        self.num_input_timestamps = num_input_timestamps
        self.num_label_timestamps = num_label_timestamps
        self.stride = stride
        self.transforms = transforms

        self.files = self._read_data(file_path)
        self.num_samples_per_day = self.files[0].shape[0]
        self.num_samples = self.num_samples_per_day * len(self.date_list)

    def _get_date_strs(self, date_period: Tuple[str, ...]) -> List:
        """Get a string list of all dates within given period.

        Args:
            date_period (Tuple[str,...]): Dates of data. Scale is [start_date, end_date] with format "%Y%m%d".
        """
        start_time = datetime.strptime(date_period[0], "%Y%m%d")
        end_time = datetime.strptime(date_period[1], "%Y%m%d")
        results = []
        current_time = start_time
        while current_time <= end_time:
            date_str = current_time.strftime("%Y%m%d")
            results.append(date_str)
            current_time += timedelta(days=1)
        return results

    def _read_data(self, path: str):
        if path.endswith(".h5"):
            paths = [path]
        else:
            paths = [
                _path
                for _path in glob.glob(osp.join(path, "*.h5"))
                if _path.split(".h5")[0].split("_")[-1] in self.date_list
            ]
        assert len(paths) == len(
            self.date_list
        ), f"Data of {len(self.date_list)} days wanted but only {len(paths)} days be found"
        paths.sort()

        files = [h5py.File(_path, "r")["dataset"] for _path in paths]
        return files

    def __len__(self):
        return (
            self.num_samples // self.stride
            - self.num_input_timestamps
            - self.num_label_timestamps
            + 1
        )

    def __getitem__(self, global_idx):
        global_idx *= self.stride
        _samples = np.empty(
            (
                self.num_input_timestamps + self.num_label_timestamps,
                *self.files[0].shape[1:],
            ),
            dtype=paddle.get_default_dtype(),
        )
        for idx in range(self.num_input_timestamps + self.num_label_timestamps):
            sample_idx = global_idx + idx * self.stride
            day_idx = sample_idx // self.num_samples_per_day
            local_idx = sample_idx % self.num_samples_per_day
            _samples[idx] = self.files[day_idx][local_idx]

        input_item = {self.input_keys[0]: _samples[: self.num_input_timestamps]}
        label_item = {self.label_keys[0]: _samples[self.num_input_timestamps :]}

        weight_shape = [1] * len(next(iter(label_item.values())).shape)
        weight_item = {
            key: np.full(weight_shape, value, paddle.get_default_dtype())
            for key, value in self.weight_dict.items()
        }

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                input_item, label_item, weight_item
            )

        return input_item, label_item, weight_item


class MRMSSampledDataset(io.Dataset):
    """Class for MRMS sampled dataset. MRMS one sample's data is stored in a .h5 file. Each file includes keys "date"/"time_interval"/"dataset".
        The class just return data by input_item and values of label_item are empty for all label_keys.

    Args:
        file_path (str): Dataset path.
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        num_total_timestamps (int, optional):  Number of timestamp of input+label. Defaults to 1.
        transforms (Optional[vision.Compose]): Composed transform functor(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.MRMSSampledDataset(
        ...     "file_path": "/path/to/MRMSSampledDataset",
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "num_total_timestamps": 29,
        ... )  # doctest: +SKIP
        >>> # get the length of the dataset
        >>> dataset_size = len(dataset)  # doctest: +SKIP
        >>> # get the first sample of the data
        >>> first_sample = dataset[0]  # doctest: +SKIP
        >>> print("First sample:", first_sample)  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        num_total_timestamps: int = 1,
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

        self.num_total_timestamps = num_total_timestamps
        self.transforms = transforms

        self.files = self._read_data(file_path)
        self.num_samples = len(self.files)

    def _read_data(self, path: str):
        paths = glob.glob(osp.join(path, "*.h5"))
        paths.sort()
        files = [h5py.File(_path, "r")["dataset"] for _path in paths]
        return files

    def __len__(self):
        return self.num_samples - self.num_total_timestamps + 1

    def __getitem__(self, global_idx):
        _samples = []
        for idx in range(global_idx, global_idx + self.num_total_timestamps):
            _samples.append(np.expand_dims(self.files[idx], axis=0))

        input_item = {
            self.input_keys[0]: np.concatenate(_samples, axis=0).astype(
                paddle.get_default_dtype()
            )
        }
        label_item = {}
        weight_item = {}

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                input_item, label_item, weight_item
            )

        return input_item, label_item, weight_item

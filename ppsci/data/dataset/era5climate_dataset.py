# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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

import numbers
import os
import random
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

try:
    import h5py
except ModuleNotFoundError:
    pass
try:
    import xarray as xr
except ModuleNotFoundError:
    pass
import numpy as np
import paddle
from paddle import io
from paddle import vision


class ERA5ClimateDataset(io.Dataset):
    """ERA5 dataset for multi-meteorological-element climate prediction (r, t, u, v).

    Args:
        file_path (str): Dataset path (contains .npy files in year folders).
        input_keys (Tuple[str, ...]): Input dict keys, e.g. ("input",).
        label_keys (Tuple[str, ...]): Label dict keys, e.g. ("output",).
        size (Tuple[int, int]): Crop size (height, width).
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        transforms (Optional[vision.Compose]): Optional transforms. Defaults to None.
        training (bool): If in training mode (2016-2018). Else validation mode (2019).
        stride (int): Stride for sampling. Defaults to 1.
        sq_length (int): Sequence length for input and output. Defaults to 6.
        years (Optional[List[str]]): List of years to load. Defaults to None (use default years).
    """

    batch_index: bool = False

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        size: Tuple[int, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        transforms: Optional[vision.Compose] = None,
        training: bool = True,
        stride: int = 1,
        sq_length: int = 6,
        years: Optional[List[str]] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.size = size
        self.training = training
        self.sq_length = sq_length
        self.transforms = transforms
        self.stride = stride
        self.group_size = 24 * 7  # 168 hours per week

        mean_file_path = os.path.join(self.file_path, "mean.nc")
        std_file_path = os.path.join(self.file_path, "std.nc")

        mean_ds = xr.open_dataset(mean_file_path)
        std_ds = xr.open_dataset(std_file_path)

        self.mean = mean_ds["mean"].values.reshape(-1, 1, 1)
        self.std = std_ds["std"].values.reshape(-1, 1, 1)

        print("Start loading all hourly data from the HDF5 file...")
        start_time = time.time()

        if self.training:
            years = ["2016", "2017", "2018"] if years is None else years
        else:
            years = ["2019"] if years is None else years

        all_hourly_data = []
        for year in years:
            h5_filepath = os.path.join(self.file_path, f"{year}.h5")
            if not os.path.exists(h5_filepath):
                raise FileNotFoundError(f"h5 file not found: {h5_filepath}")

            print(f"Loading {h5_filepath}...")
            with h5py.File(h5_filepath, "r") as hf:
                all_hourly_data.append(hf["data"][:])

        self.data_hourly = np.concatenate(all_hourly_data, axis=0)

        end_time = time.time()
        print("Data loaded!")
        print(
            f"Total hours: {self.data_hourly.shape[0]}, Shape: {self.data_hourly.shape}"
        )
        print(f"Estimated memory usage: {self.data_hourly.nbytes / 1e9:.2f} GB")
        print(f"Loading time: {end_time - start_time:.2f} seconds.")

        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

    def __len__(self):
        group_size = 24 * 7  # 7 days of hourly data
        span = 2 * self.sq_length * group_size
        return self.data_hourly.shape[0] - span + 1

    def __getitem__(self, global_idx):
        x_start_hour = global_idx
        x_end_hour = x_start_hour + self.sq_length * self.group_size

        y_start_hour = x_end_hour
        y_end_hour = y_start_hour + self.sq_length * self.group_size

        x_hourly = self.data_hourly[x_start_hour:x_end_hour]
        y_hourly = self.data_hourly[y_start_hour:y_end_hour]

        x_weekly_groups = x_hourly.reshape(
            self.sq_length, self.group_size, *x_hourly.shape[1:]
        )
        y_weekly_groups = y_hourly.reshape(
            self.sq_length, self.group_size, *y_hourly.shape[1:]
        )

        x = np.mean(x_weekly_groups, axis=1)  # x.shape: (sq_length, 12, H, W)
        y = np.mean(y_weekly_groups, axis=1)  # y.shape: (sq_length, 12, H, W)

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x, y = self._random_crop(x, y)

        input_item = {self.input_keys[0]: x.astype(np.float32)}
        label_item = {self.label_keys[0]: y.astype(np.float32)}

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

    def _random_crop(self, x, y):
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))

        th, tw = self.size
        h, w = y.shape[-2], y.shape[-1]  # Get the original height and width from y

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        x_cropped = x[..., y1 : y1 + th, x1 : x1 + tw]
        y_cropped = y[..., y1 : y1 + th, x1 : x1 + tw]

        return x_cropped, y_cropped

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

import datetime
import numbers
import os
import random
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import paddle

try:
    import xarray as xr
except ModuleNotFoundError:
    pass
from paddle import io
from paddle import vision


class ERA5MeteoDataset(io.Dataset):
    """ERA5 dataset for multi-meteorological-element prediction (r, t, u, v).

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

        mean_file_path = os.path.join(self.file_path, "mean.nc")
        std_file_path = os.path.join(self.file_path, "std.nc")

        mean_ds = xr.open_dataset(mean_file_path)
        std_ds = xr.open_dataset(std_file_path)

        self.mean = mean_ds["mean"].values.reshape(-1, 1, 1)
        self.std = std_ds["std"].values.reshape(-1, 1, 1)

        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.time_table = self._build_time_table()

    def _build_time_table(self):
        """Build datetime list from available .npy files, filtered by years."""
        years = sorted([y for y in os.listdir(self.file_path) if y.isdigit()])

        if self.training:
            target_years = {"2016", "2017", "2018"}
        else:
            target_years = {"2016", "2019"}

        time_list = []
        for y in years:
            if y not in target_years:
                continue
            year_dir = os.path.join(self.file_path, y)
            files = sorted(os.listdir(year_dir))
            for fname in files:
                if fname.startswith("r_") and fname.endswith(".npy"):
                    dt_str = fname[2:12]  # YYYYMMDDHH
                    dt = datetime.datetime.strptime(dt_str, "%Y%m%d%H")
                    time_list.append(dt)

        return sorted(time_list)

    def __len__(self):
        return len(self.time_table) - self.sq_length * 2 + 1

    def __getitem__(self, global_idx):
        x_list, y_list = [], []

        for m in range(self.sq_length):
            x_list.append(self.load_data(global_idx + m))

        for n in range(self.sq_length):
            y_list.append(self.load_data(global_idx + self.sq_length + n))

        x = np.stack(x_list, axis=0)
        y = np.stack(y_list, axis=0)

        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x, y = self._random_crop(x, y)

        input_item = {self.input_keys[0]: x}
        label_item = {self.label_keys[0]: y}

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

    def load_data(self, indices):
        """Load r, t, u, v for a given index."""
        dt = self.time_table[indices]
        year = f"{dt.year:04d}"
        mon = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        hour = f"{dt.hour:02d}"

        r_data = np.load(
            os.path.join(self.file_path, year, f"r_{year}{mon}{day}{hour}.npy")
        )
        t_data = np.load(
            os.path.join(self.file_path, year, f"t_{year}{mon}{day}{hour}.npy")
        )
        u_data = np.load(
            os.path.join(self.file_path, year, f"u_{year}{mon}{day}{hour}.npy")
        )
        v_data = np.load(
            os.path.join(self.file_path, year, f"v_{year}{mon}{day}{hour}.npy")
        )

        data = np.concatenate([r_data, t_data, u_data, v_data])
        return data

    def _random_crop(self, x, y):
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))

        th, tw = self.size
        h, w = y.shape[-2], y.shape[-1]

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        x_cropped = x[..., y1 : y1 + th, x1 : x1 + tw]
        y_cropped = y[..., y1 : y1 + th, x1 : x1 + tw]

        return x_cropped, y_cropped

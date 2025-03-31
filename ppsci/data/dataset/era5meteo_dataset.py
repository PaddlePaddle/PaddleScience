# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

import h5py
import numpy as np
import paddle
from paddle import io
from paddle import vision


class ERA5MeteoDataset(io.Dataset):
    """Class for ERA5 dataset.

    Args:
        file_path (str): Dataset path.
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
        >>> dataset = ppsci.data.dataset.ERA5MeteoDataset(
        ...     "file_path": "/path/to/ERA5MeteoDataset",
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "size": ("H", "W"),
        ... )  # doctest: +SKIP
    """

    # Whether support batch indexing for speeding up fetching process.
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
        time_step: int = 6,
    ):
        super().__init__()
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.size = size
        self.training = training
        self.sq_length = sq_length

        self.time_step = time_step

        self.transforms = transforms

        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        # load precipitation data
        if training:
            self.precipitation = h5py.File(
                os.path.join(self.file_path, "train", "rain_2016_01.h5")
            )
        else:
            self.precipitation = h5py.File(
                os.path.join(self.file_path, "test", "rain_2020_02.h5")
            )

        t_list = self.precipitation["time"][:]
        start_time = datetime.datetime(1900, 1, 1, 0, 0, 0)
        self.time_table = []
        for i in range(len(t_list)):
            temp = start_time + datetime.timedelta(hours=int(t_list[i]))
            self.time_table.append(temp)

    def __len__(self):
        return len(self.time_table) - 24

    def __getitem__(self, global_idx):

        x, y_t, y_r, y_u, y_v = [], [], [], [], []

        for m in range(self.sq_length):
            x.append(self.load_data(global_idx + m * self.time_step))
        for n in range(self.sq_length):
            future_data = self.load_data(global_idx + (self.sq_length + n) * self.time_step)
            y_t.append(future_data[1])  # Temperature
            y_r.append(future_data[0])  # Humidity
            y_u.append(future_data[2])  # U-Wind
            y_v.append(future_data[3])  # V-Wind

        x, y_t, y_r, y_u, y_v = self._random_crop(x, y_t, y_r, y_u, y_v)

        input_item = {self.input_keys[0]: np.stack(x, axis=0)}
        label_item = {self.label_keys[0]: np.stack([y_t, y_r, y_u, y_v], axis=0)}

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
        year = str(self.time_table[indices].timetuple().tm_year)
        mon = str(self.time_table[indices].timetuple().tm_mon)
        if len(mon) == 1:
            mon = "0" + mon
        day = str(self.time_table[indices].timetuple().tm_mday)
        if len(day) == 1:
            day = "0" + day
        hour = str(self.time_table[indices].timetuple().tm_hour)
        if len(hour) == 1:
            hour = "0" + hour
        r_data = np.load(os.path.join(self.file_path, year, f"r_{year}{mon}{day}{hour}.npy"))
        t_data = np.load(os.path.join(self.file_path, year, f"t_{year}{mon}{day}{hour}.npy"))
        u_data = np.load(os.path.join(self.file_path, year, f"u_{year}{mon}{day}{hour}.npy"))
        v_data = np.load(os.path.join(self.file_path, year, f"v_{year}{mon}{day}{hour}.npy"))

        data = np.concatenate([r_data, t_data, u_data, v_data])

        return data

    def _random_crop(self, x, y_t, y_r, y_u, y_v):
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        th, tw = self.size
        h, w = y_t[0].shape[-2], y_t[0].shape[-1]
        x1, y1 = random.randint(0, w - tw), random.randint(0, h - th)

        # Apply cropping
        x = [self._crop(xi, y1, x1, y1 + th, x1 + tw) for xi in x]
        y_t = [self._crop(y, y1, x1, y1 + th, x1 + tw) for y in y_t]
        y_r = [self._crop(y, y1, x1, y1 + th, x1 + tw) for y in y_r]
        y_u = [self._crop(y, y1, x1, y1 + th, x1 + tw) for y in y_u]
        y_v = [self._crop(y, y1, x1, y1 + th, x1 + tw) for y in y_v]

        return x, y_t, y_r, y_u, y_v

    def _crop(self, im, x_start, y_start, x_end, y_end):
        if len(im.shape) == 3:
            return im[:, x_start:x_end, y_start:y_end]
        else:
            return im[x_start:x_end, y_start:y_end]

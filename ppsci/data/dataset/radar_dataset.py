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

import os
from typing import Dict
from typing import Optional
from typing import Tuple

try:
    import cv2
except ModuleNotFoundError:
    pass

import importlib

import numpy as np
import paddle
from paddle import io


class RadarDataset(io.Dataset):
    """Class for Radar dataset.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        image_width (int): Image width.
        image_height (int): Image height.
        total_length (int): Total length.
        dataset_path (str): Dataset path.
        data_type (str): Input and output data type. Defaults to paddle.get_default_dtype().
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.RadarDataset(
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "image_width": 512,
        ...     "image_height": 512,
        ...     "total_length": 29,
        ...     "dataset_path": "datasets/mrms/figure",
        ...     "data_type": paddle.get_default_dtype(),
        ... )  # doctest: +SKIP
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        image_width: int,
        image_height: int,
        total_length: int,
        dataset_path: str,
        data_type: str = paddle.get_default_dtype(),
        weight_dict: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if importlib.util.find_spec("cv2") is None:
            raise ModuleNotFoundError(
                "To use RadarDataset, please install 'opencv-python' via: `pip install "
                "opencv-python` first."
            )
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.img_width = image_width
        self.img_height = image_height
        self.length = total_length
        self.dataset_path = dataset_path
        self.data_type = data_type

        self.weight_dict = {} if weight_dict is None else weight_dict
        if weight_dict is not None:
            self.weight_dict = {key: 1.0 for key in self.label_keys}
            self.weight_dict.update(weight_dict)

        self.case_list = []
        name_list = os.listdir(self.dataset_path)
        name_list.sort()
        for name in name_list:
            case = []
            for i in range(29):
                case.append(
                    self.dataset_path
                    + "/"
                    + name
                    + "/"
                    + name
                    + "-"
                    + str(i).zfill(2)
                    + ".png"
                )
            self.case_list.append(case)

    def _load(self, index):
        data = []
        for img_path in self.case_list[index]:
            img = cv2.imread(img_path, 2)
            data.append(np.expand_dims(img, axis=0))
        data = np.concatenate(data, axis=0).astype(self.data_type) / 10.0 - 3.0
        assert data.shape[1] <= 1024 and data.shape[2] <= 1024
        return data

    def __getitem__(self, index):
        data = self._load(index)[-self.length :].copy()
        mask = np.ones_like(data)
        mask[data < 0] = 0
        data[data < 0] = 0
        data = np.clip(data, 0, 128)
        vid = np.zeros(
            (self.length, self.img_height, self.img_width, 2), dtype=self.data_type
        )
        vid[..., 0] = data
        vid[..., 1] = mask

        input_item = {self.input_keys[0]: vid}
        label_item = {}
        weight_item = {}
        for key in self.label_keys:
            label_item[key] = np.asarray([], paddle.get_default_dtype())
        if len(label_item) > 0:
            weight_shape = [1] * len(next(iter(label_item.values())).shape)
            weight_item = {
                key: np.full(weight_shape, value, paddle.get_default_dtype())
                for key, value in self.weight_dict.items()
            }
        return input_item, label_item, weight_item

    def __len__(self):
        return len(self.case_list)

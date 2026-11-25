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

from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np


class Translate:
    """Translate class.

    Args:
        offset (Dict[str, float]): Shift the input data according to the variable name
            and coefficient specified in offset.

    Examples:
        >>> import ppsci
        >>> translate = ppsci.data.transform.Translate({"x": 1.0, "y": -1.0})
    """

    def __init__(self, offset: Dict[str, float]):
        self.offset = offset

    def __call__(self, data_dict):
        for key in self.offset:
            if key in data_dict:
                data_dict[key] += self.offset[key]
        return data_dict


class Scale:
    """Scale class.

    Args:
        scale (Dict[str, float]): Scale the input data according to the variable name
            and coefficient specified in scale.

    Examples:
        >>> import ppsci
        >>> translate = ppsci.data.transform.Scale({"x": 1.5, "y": 2.0})
    """

    def __init__(self, scale: Dict[str, float]):
        self.scale = scale

    def __call__(self, data_dict):
        for key in self.scale:
            if key in data_dict:
                data_dict[key] *= self.scale[key]
        return data_dict


class Normalize:
    """Normalize data class.

    Args:
        mean (Union[np.array, Tuple[float, ...]]): Mean of training dataset.
        std (Union[np.array, Tuple[float, ...]]): Standard Deviation of training dataset.
        apply_keys (Tuple[str, ...], optional): Which data is the normalization method applied to. Defaults to ("input", "label").

    Examples:
        >>> import ppsci
        >>> normalize = ppsci.data.transform.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    """

    def __init__(
        self,
        mean: Union[np.array, Tuple[float, ...]],
        std: Union[np.array, Tuple[float, ...]],
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        if len(apply_keys) == 0 or len(set(apply_keys) | {"input", "label"}) > 2:
            raise ValueError(
                f"apply_keys should be a non empty subset of ('input', 'label'), but got {apply_keys}"
            )
        self.mean = mean
        self.std = std
        self.apply_keys = apply_keys

    def __call__(self, data):
        input_item, label_item, weight_item = data
        if "input" in self.apply_keys:
            for key, value in input_item.items():
                input_item[key] = (value - self.mean) / self.std
        if "label" in self.apply_keys:
            for key, value in label_item.items():
                label_item[key] = (value - self.mean) / self.std
        return input_item, label_item, weight_item


class Log1p:
    """Calculates the natural logarithm of one plus the data, element-wise.

    Args:
        scale (float, optional): Scale data. Defaults to 1.0.
        apply_keys (Tuple[str, ...], optional): Which data is the log1p method applied to. Defaults to ("input", "label").

    Examples:
        >>> import ppsci
        >>> log1p = ppsci.data.transform.Log1p(1e-5)
    """

    def __init__(
        self,
        scale: float = 1.0,
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        if len(apply_keys) == 0 or len(set(apply_keys) | {"input", "label"}) > 2:
            raise ValueError(
                f"apply_keys should be a non empty subset of ('input', 'label'), but got {apply_keys}"
            )
        self.scale = scale
        self.apply_keys = apply_keys

    def __call__(self, data):
        input_item, label_item, weight_item = data
        if "input" in self.apply_keys:
            for key, value in input_item.items():
                input_item[key] = np.log1p(value / self.scale)
        if "label" in self.apply_keys:
            for key, value in label_item.items():
                label_item[key] = np.log1p(value / self.scale)
        return input_item, label_item, weight_item


class CropData:
    """Crop data class.

    Args:
        xmin (Tuple[int, ...]): Bottom left corner point, [x0, y0].
        xmax (Tuple[int, ...]): Top right corner point, [x1, y1].
        apply_keys (Tuple[str, ...], optional): Which data is the crop method applied to. Defaults to ("input", "label").

    Examples:
        >>> import ppsci
        >>> crop_data = ppsci.data.transform.CropData((0, 0), (720, 1440))
    """

    def __init__(
        self,
        xmin: Tuple[int, ...],
        xmax: Tuple[int, ...],
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        if len(apply_keys) == 0 or len(set(apply_keys) | {"input", "label"}) > 2:
            raise ValueError(
                f"apply_keys should be a non empty subset of ('input', 'label'), but got {apply_keys}"
            )
        self.xmin = xmin
        self.xmax = xmax
        self.apply_keys = apply_keys

    def __call__(self, data):
        input_item, label_item, weight_item = data
        if "input" in self.apply_keys:
            for key, value in input_item.items():
                input_item[key] = value[
                    :, self.xmin[0] : self.xmax[0], self.xmin[1] : self.xmax[1]
                ]
        if "label" in self.apply_keys:
            for key, value in label_item.items():
                label_item[key] = value[
                    :, self.xmin[0] : self.xmax[0], self.xmin[1] : self.xmax[1]
                ]
        return input_item, label_item, weight_item


class SqueezeData:
    """Squeeze data clsss.

    Args:
        apply_keys (Tuple[str, ...], optional): Which data is the squeeze method applied to. Defaults to ("input", "label").

    Examples:
        >>> import ppsci
        >>> squeeze_data = ppsci.data.transform.SqueezeData()
    """

    def __init__(self, apply_keys: Tuple[str, ...] = ("input", "label")):
        if len(apply_keys) == 0 or len(set(apply_keys) | {"input", "label"}) > 2:
            raise ValueError(
                f"apply_keys should be a non empty subset of ('input', 'label'), but got {apply_keys}"
            )
        self.apply_keys = apply_keys

    def __call__(self, data):
        input_item, label_item, weight_item = data
        if "input" in self.apply_keys:
            for key, value in input_item.items():
                if value.ndim == 4:
                    B, C, H, W = value.shape
                    input_item[key] = value.reshape((B * C, H, W))
                if value.ndim != 3:
                    raise ValueError(
                        f"Only support squeeze data to ndim=3 now, but got ndim={value.ndim}"
                    )
        if "label" in self.apply_keys:
            for key, value in label_item.items():
                if value.ndim == 4:
                    B, C, H, W = value.shape
                    label_item[key] = value.reshape((B * C, H, W))
                if value.ndim != 3:
                    raise ValueError(
                        f"Only support squeeze data to ndim=3 now, but got ndim={value.ndim}"
                    )
        return input_item, label_item, weight_item

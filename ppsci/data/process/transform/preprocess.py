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
from typing import Union


class Translate:
    """Translate class, a transform mainly for mesh.

    Args:
        offset (Dict[str, float]): Shift the input data according to the variable name
            and coefficient specified in offset.

    Examples:
        >>> import ppsci
        >>> translate = ppsci.data.transform.Translate({"x": 1.0, "y": -1.0})
    """

    def __init__(self, offset: Dict[str, Union[int, float]]):
        self.offset = offset

    def __call__(self, data_dict):
        for key in self.offset:
            data_dict[key] += self.offset[key]
        return data_dict


class Scale:
    """Scale class, a transform mainly for mesh.

    Args:
        scale (Dict[str, float]): Scale the input data according to the variable name
            and coefficient specified in scale.

    Examples:
        >>> import ppsci
        >>> translate = ppsci.data.transform.Scale({"x": 1.5, "y": 2.0})
    """

    def __init__(self, scale: Dict[str, Union[int, float]]):
        self.scale = scale

    def __call__(self, data_dict):
        for key in self.scale:
            if key in data_dict:
                data_dict[key] *= self.scale[key]
        return data_dict

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

import copy
import traceback
from typing import Any
from typing import Tuple

from paddle import vision

from ppsci.data.process.transform.preprocess import CropData
from ppsci.data.process.transform.preprocess import FunctionalTransform
from ppsci.data.process.transform.preprocess import Log1p
from ppsci.data.process.transform.preprocess import Normalize
from ppsci.data.process.transform.preprocess import Scale
from ppsci.data.process.transform.preprocess import SqueezeData
from ppsci.data.process.transform.preprocess import Translate

__all__ = [
    "CropData",
    "FunctionalTransform",
    "Log1p",
    "Normalize",
    "Scale",
    "SqueezeData",
    "Translate",
    "build_transforms",
]


class Compose(vision.Compose):
    """Custom Compose for multiple items in given data."""

    def __call__(self, *data: Tuple[Any, ...]):
        for f in self.transforms:
            try:
                # NOTE: This is different from vision.Compose to allow receive multiple data items
                data = f(*data)
            except Exception as e:
                stack_info = traceback.format_exc()
                print(
                    f"fail to perform transform [{f}] with error: "
                    f"{e} and stack:\n{str(stack_info)}"
                )
                raise e
        return data


def build_transforms(cfg):
    if not cfg:
        return Compose([])
    cfg = copy.deepcopy(cfg)

    transform_list = []
    for _item in cfg:
        transform_cls = next(iter(_item.keys()))
        transform_cfg = _item[transform_cls]
        transform = eval(transform_cls)(**transform_cfg)
        transform_list.append(transform)

    return Compose(transform_list)

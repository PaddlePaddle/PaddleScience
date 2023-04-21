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
from typing import Any
from typing import List

from ppsci.data import _default_collate_fn_allow_none
from ppsci.data.process import transform

__all__ = ["build_batch_transforms"]


def build_batch_transforms(cfg):
    cfg = copy.deepcopy(cfg)
    batch_transforms = transform.build_transforms(cfg)

    def collate_fn_batch_transforms(batch: List[Any]):
        # apply batch transform on uncollated data
        batch = batch_transforms(batch)
        # then do collate
        return _default_collate_fn_allow_none(batch)

    return collate_fn_batch_transforms

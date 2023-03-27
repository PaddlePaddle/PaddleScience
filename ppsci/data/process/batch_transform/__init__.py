"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
from typing import Any
from typing import List

import numpy as np

from ppsci.data.process import transform

__all__ = ["build_batch_transforms"]


def build_batch_transforms(cfg):
    cfg = copy.deepcopy(cfg)
    batch_transforms = transform.build_transforms(cfg)

    def collate_fn_batch_transforms(batch_data_list: List[Any]):
        # batch_data_list: [(Di, Dl, Dw), (Di, Dl, Dw), ..., (Di, Dl, Dw)]
        batch_data_list = batch_transforms(batch_data_list)

        # batch each field
        collated_data = []
        num_components = len(batch_data_list[0])

        # compose batch for every component
        for component_idx in range(num_components):
            # [Dx, Dx, ..., Dx]
            component_list = [batch[component_idx] for batch in batch_data_list]
            batched_component = {}
            # compose each key in current component
            for key in component_list[0]:
                batched_component[key] = np.stack(
                    [sample[key] for sample in component_list], axis=0
                )
            collated_data.append(batched_component)
        return collated_data

    return collate_fn_batch_transforms

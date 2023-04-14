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

import numpy as np

from ppsci.data import dataset
from ppsci.utils import misc
from ppsci.validate import base


class DataValidator(base.Validator):
    def __init__(
        self,
        data_dict,
        input_keys,
        label_keys,
        alias_dict,
        dataloader_cfg,
        loss,
        transforms=None,
        metric=None,
        name=None,
    ):
        # read data
        self.input_keys = [
            alias_dict[key] if key in alias_dict else key for key in input_keys
        ]
        self.label_keys = [
            alias_dict[key] if key in alias_dict else key for key in label_keys
        ]

        input = {}
        for key in self.input_keys:
            input[key] = data_dict[key]

        label = {}
        for key in self.label_keys:
            label[key] = data_dict[key]

        self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.label_keys}
        self.num_timestamp = 1

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(
            input, label, weight, transforms
        )

        super().__init__(_dataset, dataloader_cfg, loss, metric, name)

    def __str__(self):
        return ", ".join(
            [
                self.__class__.__name__,
                f"name = {self.name}",
                f"input_keys = {self.input_keys}",
                f"label_keys = {self.label_keys}",
                f"len(dataloader) = {len(self.data_loader.dataset)}",
                f"loss = {self.loss}",
                f"metric = {list(self.metric.keys())}",
            ]
        )

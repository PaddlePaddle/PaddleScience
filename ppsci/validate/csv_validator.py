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

import types

import numpy as np

from ppsci.data import dataset
from ppsci.utils import misc
from ppsci.validate import base


class CSVValidator(base.Validator):
    """Validator for csv file

    Args:
        csv_path (str): CSV file path.
        input_keys (List[str]): Input keys in csv file, such as ["X:0", "X:1"].
        label_keys (List[str]): Label keys in csv file, such as ["U:0", "U:1"].
        alias_dict (Dict[str, str]): Alias name for input/label keys, such as
            {"X:0": "x", "X:1": "y", "U:0": "u", "U:1": "v"}.
        dataloader_cfg (Dict): Config of building a dataloader
        loss (LossBase): Loss functor.
        transforms (vision.Compose): Composed transforms.
        metric (Dict[str, Metric], optional): Named metric functors in dict.
            Defaults to None.
        name (str, optional): Name of validator. Defaults to None.
    """

    def __init__(
        self,
        csv_path,
        input_keys,
        label_keys,
        alias_dict,
        dataloader_cfg,
        loss,
        transforms=None,
        metric=None,
        weight_dict=None,
        name=None,
    ):
        # read data
        data_dict = misc.load_csv_file(csv_path, input_keys + label_keys, alias_dict)
        self.input_keys = [
            alias_dict[key] if key in alias_dict else key for key in input_keys
        ]
        self.output_keys = [
            alias_dict[key] if key in alias_dict else key for key in input_keys
        ]

        input = {}
        for key in self.input_keys:
            input[key] = data_dict[key]

        label = {}
        for key in self.output_keys:
            label[key] = data_dict[key]

        self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
        self.num_timestamp = 1

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), float(value))
                elif isinstance(value, types.FunctionType):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), float(weight[key])
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

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
                f"output_keys = {self.output_keys}",
                f"len(dataloader) = {len(self.data_loader.dataset)}",
                f"loss = {self.loss}",
                f"metric = {list(self.metric.keys())}",
            ]
        )

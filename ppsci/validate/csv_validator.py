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

import os.path as osp

import numpy as np
import pandas as pd

from ppsci.data import dataset
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
        name=None,
    ):
        if not osp.exists(csv_path):
            raise FileNotFoundError(f"csv_path({csv_path}) not exist.")

        # read data
        raw_data_frame = pd.read_csv(csv_path)

        # convert to numpy array
        input = {}
        for key in input_keys:
            input[key] = np.asarray(raw_data_frame[key], "float32")
            input[key] = input[key].reshape([-1, 1])
        label = {}
        for key in label_keys:
            label[key] = np.asarray(raw_data_frame[key], "float32")
            label[key] = label[key].reshape([-1, 1])

        # replace key with alias
        for key, alias in alias_dict.items():
            if key in input_keys:
                input[alias] = input.pop(key)
            elif key in label_keys:
                label[alias] = label.pop(key)
            else:
                raise ValueError(
                    f"key({key}) in alias_dict didn't appear "
                    f"in input_keys or label_keys"
                )
        self.input_keys = list(input.keys())
        self.output_keys = list(label.keys())
        self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}
        self.num_timestamp = 1 if "t" not in input else len(np.unique(input["t"]))

        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        _dataset = getattr(dataset, dataloader_cfg["dataset"])(
            input, label, weight, transforms
        )

        super().__init__(_dataset, dataloader_cfg, loss, metric, name)

    def __str__(self):
        _str = ", ".join(
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
        return _str

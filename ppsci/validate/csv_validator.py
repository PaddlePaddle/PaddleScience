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

from ppsci.data import dataset
from ppsci.validate import base


class CSVValidator(base.Validator):
    """Validator for csv file

    Args:
        file_path (str): CSV file path.
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
        label_expr,
        dataloader_cfg,
        loss,
        metric=None,
        name=None,
    ):
        self.label_expr = label_expr

        # build dataset
        _dataset = dataset.build_dataset(dataloader_cfg["dataset"])

        self.input_keys = _dataset.input_keys
        self.input_keys = _dataset.input_keys
        self.output_keys = list(label_expr.keys())
        if self.output_keys != _dataset.label_keys:
            raise ValueError(
                f"keys of label_expr({self.output_keys}) "
                f"should be same as _dataset.label_keys({_dataset.label_keys})"
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

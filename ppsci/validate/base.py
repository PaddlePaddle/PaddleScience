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

from typing import Any
from typing import Dict

from paddle import io

from ppsci import data
from ppsci import loss
from ppsci import metric


class Validator:
    """Base class for validators.

    Args:
        dataset (io.Dataset): Dataset for validator.
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        loss (loss.Loss): Loss functor.
        metric (Dict[str, metric.Metric]): Named metric functors in dict.
        name (str): Name of validator.
    """

    def __init__(
        self,
        dataset: io.Dataset,
        dataloader_cfg: Dict[str, Any],
        loss: loss.Loss,
        metric: Dict[str, metric.Metric],
        name: str,
    ):
        self.data_loader = data.build_dataloader(dataset, dataloader_cfg)
        self.data_iter = iter(self.data_loader)
        self.loss = loss
        self.metric = metric
        self.name = name

    def __str__(self):
        return ", ".join(
            [
                self.__class__.__name__,
                f"name = {self.name}",
                f"input_keys = {self.input_keys}",
                f"output_keys = {self.output_keys}",
                f"output_expr = {self.output_expr}",
                f"label_dict = {self.label_dict}",
                f"len(dataloader) = {len(self.data_loader)}",
                f"loss = {self.loss}",
                f"metric = {list(self.metric.keys())}",
            ]
        )

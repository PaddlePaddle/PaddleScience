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
from typing import Callable
from typing import Dict
from typing import Optional

from ppsci import loss
from ppsci import metric
from ppsci.data import dataset
from ppsci.validate import base


class CSVValidator(base.Validator):
    """Validator for csv file

    Args:
        dataloader_cfg (Dict[str, Any]): Config of building a dataloader
        loss (loss.LossBase): Loss functor.
        label_expr (Optional[Dict[str, Callable]]): Function in dict for computing output.
            e.g. {"u_mul_v": lambda out: out["u"] * out["v"]} means the model output u
            will be multiplied by model output v and the result will be named "u_mul_v".
        metric (Optional[Dict[str, metric.MetricBase]], optional): Named metric functors in dict.
            Defaults to None.
        name (str, optional): Name of validator. Defaults to None.
    """

    def __init__(
        self,
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        label_expr: Optional[Dict[str, Callable]] = None,
        metric: Optional[Dict[str, metric.MetricBase]] = None,
        name: str = None,
    ):
        self.label_expr = label_expr

        # build dataset
        _dataset = dataset.build_dataset(dataloader_cfg["dataset"])

        self.input_keys = _dataset.input_keys
        self.output_keys = (
            list(label_expr.keys()) if label_expr is not None else _dataset.output_keys
        )

        if self.label_expr is None:
            self.label_expr = {
                key: lambda out, k=key: out[k] for key in self.output_keys
            }

        # construct dataloader with dataset and dataloader_cfg
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

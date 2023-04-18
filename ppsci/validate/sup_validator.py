# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any
from typing import Callable
from typing import Dict

from ppsci import loss
from ppsci.data import dataset
from ppsci.validate import base


class SupervisedValidator(base.Validator):
    """Validator for supervised models.

    Args:
        label_expr (Dict[str, Callable]): List of label expression.
        dataloader_cfg (Dict[str, Any]): Config of building a dataloader.
        loss (loss.LossBase): Loss functor.
        metric (Dict[str, Any], optional): Named metric functors in dict. Defaults to None.
        name (str, optional): Name of validator. Defaults to None.
    """

    def __init__(
        self,
        label_expr: Dict[str, Callable],
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        metric: Dict[str, Any] = None,
        name: str = None,
    ):
        self.label_expr = label_expr

        # build dataset
        _dataset = dataset.build_dataset(dataloader_cfg["dataset"])

        self.input_keys = _dataset.input_keys
        self.output_keys = list(label_expr.keys())

        if self.output_keys != _dataset.label_keys:
            raise ValueError(
                f"keys of label_expr({self.output_keys}) "
                f"should be same as _dataset.label_keys({_dataset.label_keys})"
            )

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, metric, name)

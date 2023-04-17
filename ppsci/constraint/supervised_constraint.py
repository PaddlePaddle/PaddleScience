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

from ppsci import loss
from ppsci.constraint import base
from ppsci.data import dataset


class SupervisedConstraint(base.Constraint):
    """Class for supervised constraint.

    Args:
        data_file (str): File path of data.
        label_expr (Dict[str, Callable]): List of label expression.
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        loss (loss.LossBase): Loss functor.
        name (str, optional): Name of constraint object. Defaults to "Sup".
    """

    def __init__(
        self,
        label_expr: Dict[str, Callable],
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        name: str = "Sup",
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
        super().__init__(_dataset, dataloader_cfg, loss, name)

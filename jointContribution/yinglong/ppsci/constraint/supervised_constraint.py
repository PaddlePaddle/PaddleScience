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
from ppsci.constraint import base
from ppsci.data import dataset


class SupervisedConstraint(base.Constraint):
    """Class for supervised constraint.

    Args:
        dataloader_cfg (Dict[str, Any]): Dataloader config.
        loss (loss.Loss): Loss functor.
        output_expr (Optional[Dict[str, Callable]]): List of label expression.
            Defaults to None.
        name (str, optional): Name of constraint object. Defaults to "Sup".

    Examples:
        >>> import ppsci
        >>> bc_sup = ppsci.constraint.SupervisedConstraint(
        ...     {
        ...         "dataset": {
        ...             "name": "IterableCSVDataset",
        ...             "file_path": "/path/to/file.csv",
        ...             "input_keys": ("x", "y"),
        ...             "label_keys": ("u", "v"),
        ...         },
        ...     },
        ...     ppsci.loss.MSELoss("mean"),
        ...     name="bc_sup",
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        dataloader_cfg: Dict[str, Any],
        loss: loss.Loss,
        output_expr: Optional[Dict[str, Callable]] = None,
        name: str = "Sup",
    ):
        self.output_expr = output_expr

        # build dataset
        _dataset = dataset.build_dataset(dataloader_cfg["dataset"])

        self.input_keys = _dataset.input_keys
        self.output_keys = (
            list(output_expr.keys()) if output_expr is not None else _dataset.label_keys
        )

        if self.output_expr is None:
            self.output_expr = {
                key: lambda out, k=key: out[k] for key in self.output_keys
            }

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, name)

    def __str__(self):
        return ", ".join(
            [
                self.__class__.__name__,
                f"name = {self.name}",
                f"input_keys = {self.input_keys}",
                f"output_keys = {self.output_keys}",
                f"output_expr = {self.output_expr}",
                f"loss = {self.loss}",
            ]
        )

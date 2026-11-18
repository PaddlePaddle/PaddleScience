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

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from ppsci import loss
from ppsci import metric
from ppsci.data import dataset
from ppsci.validate import base


class SupervisedValidator(base.Validator):
    """Validator for supervised models.

    Args:
        dataloader_cfg (Dict[str, Any]): Config of building a dataloader.
        loss (loss.Loss): Loss functor.
        output_expr (Optional[Dict[str, Callable]]): List of label expression.
        metric (Optional[Dict[str, metric.Metric]]): Named metric functors in dict. Defaults to None.
        name (Optional[str]): Name of validator. Defaults to None.

    Examples:
        >>> import ppsci
        >>> valida_dataloader_cfg = {
        ...     "dataset": {
        ...         "name": "MatDataset",
        ...         "file_path": "/path/to/file.mat",
        ...         "input_keys": ("t_f",),
        ...         "label_keys": ("eta", "f"),
        ...     },
        ...     "batch_size": 32,
        ...     "sampler": {
        ...         "name": "BatchSampler",
        ...         "drop_last": False,
        ...         "shuffle": False,
        ...     },
        ... }  # doctest: +SKIP
        >>> eta_mse_validator = ppsci.validate.SupervisedValidator(
        ...     valida_dataloader_cfg,
        ...     ppsci.loss.MSELoss("mean"),
        ...     {"eta": lambda out: out["eta"]},
        ...     metric={"MSE": ppsci.metric.MSE()},
        ...     name="eta_mse",
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        dataloader_cfg: Dict[str, Any],
        loss: loss.Loss,
        output_expr: Optional[Dict[str, Callable]] = None,
        metric: Optional[Dict[str, metric.Metric]] = None,
        name: Optional[str] = None,
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
        super().__init__(_dataset, dataloader_cfg, loss, metric, name)

    def __str__(self):
        return ", ".join(
            [
                self.__class__.__name__,
                f"name = {self.name}",
                f"input_keys = {self.input_keys}",
                f"output_keys = {self.output_keys}",
                f"output_expr = {self.output_expr}",
                f"len(dataloader) = {len(self.data_loader)}",
                f"loss = {self.loss}",
                f"metric = {list(self.metric.keys())}",
            ]
        )

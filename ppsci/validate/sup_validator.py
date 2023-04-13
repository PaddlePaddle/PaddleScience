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
from typing import Dict
from typing import Tuple

from ppsci import loss
from ppsci.data import dataset
from ppsci.validate import base


class SupervisedValidator(base.Validator):
    """Validator for supervised models.

    Args:
        input_keys (Tuple[str, ...]): Input keys.
        label_keys (Tuple[str, ...]): Output keys.
        dataloader_cfg (Dict[str, Any]): Config of building a dataloader.
        loss (loss.LossBase): Loss functor.
        metric (Dict[str, Any], optional): Named metric functors in dict. Defaults to None.
        weight_dict (Dict[str, float], optional): Weight dictionary. Defaults to None.
        name (str, optional): Name of validator. Defaults to None.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        dataloader_cfg: Dict[str, Any],
        loss: loss.LossBase,
        metric: Dict[str, Any] = None,
        weight_dict: Dict[str, float] = None,
        name: str = None,
    ):
        self.input_keys = input_keys
        self.output_keys = label_keys

        dataset_cfg = copy.deepcopy(dataloader_cfg["dataset"])
        dataset_name = dataset_cfg.pop("name")
        dataset_cfg["input_keys"] = input_keys
        dataset_cfg["label_keys"] = label_keys
        dataset_cfg["weight_dict"] = weight_dict
        _dataset = getattr(dataset, dataset_name)(**dataset_cfg)
        self.label_expr = {key: (lambda d, k=key: d[k]) for key in self.output_keys}

        super().__init__(_dataset, dataloader_cfg, loss, metric, name)

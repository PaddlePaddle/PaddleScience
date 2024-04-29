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

import copy

from ppsci.loss.mtl.agda import AGDA
from ppsci.loss.mtl.base import LossAggregator
from ppsci.loss.mtl.grad_norm import GradNorm
from ppsci.loss.mtl.pcgrad import PCGrad
from ppsci.loss.mtl.relobralo import Relobralo
from ppsci.loss.mtl.sum import Sum

__all__ = [
    "AGDA",
    "GradNorm",
    "LossAggregator",
    "PCGrad",
    "Relobralo",
    "Sum",
]


def build_mtl_aggregator(cfg):
    """Build loss aggregator with multi-task learning method.

    Args:
        cfg (AttrDict): Aggregator config.
    Returns:
        Loss: Callable loss aggregator object.
    """
    cfg = copy.deepcopy(cfg)

    aggregator_cls = cfg.pop("name")
    aggregator = eval(aggregator_cls)(**cfg)
    return aggregator

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

from ppsci.loss.mtl.agda import AGDA
from ppsci.loss.mtl.base import LossAggregator
from ppsci.loss.mtl.pcgrad import PCGrad

__all__ = [
    "LossAggregator",
    "PCGrad",
    "AGDA",
]


def build_mtl_aggregator(cfg: Dict[str, Any]):
    mtl_name = cfg.pop("name")
    return eval(mtl_name)(**cfg)

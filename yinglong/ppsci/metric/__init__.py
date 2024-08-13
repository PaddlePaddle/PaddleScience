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

from ppsci.metric.anomaly_coef import LatitudeWeightedACC
from ppsci.metric.base import Metric
from ppsci.metric.l2_rel import L2Rel
from ppsci.metric.mae import MAE
from ppsci.metric.mse import MSE
from ppsci.metric.rmse import RMSE
from ppsci.metric.rmse import LatitudeWeightedRMSE
from ppsci.utils import misc

__all__ = [
    "LatitudeWeightedACC",
    "Metric",
    "L2Rel",
    "MAE",
    "MSE",
    "RMSE",
    "LatitudeWeightedRMSE",
    "build_metric",
]


def build_metric(cfg):
    """Build metric.

    Args:
        cfg (List[AttrDict]): List of metric config.

    Returns:
        Dict[str, Metric]: Dict of callable metric object.
    """
    cfg = copy.deepcopy(cfg)

    metric_dict = misc.PrettyOrderedDict()
    for _item in cfg:
        metric_cls = next(iter(_item.keys()))
        metric_cfg = _item.pop(metric_cls)
        metric = eval(metric_cls)(**metric_cfg)
        metric_dict[metric_cls] = metric
    return metric_dict

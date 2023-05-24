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


from typing import Callable
from typing import Optional

from ppsci.metric import base


class FunctionalMetric(base.Metric):
    r"""Class for functional metric.

    Args:
        metric_expr (Optional[Callable], optional): expression of metric calculation. Defaults to None.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> def metric_expr(output_dict):
        ...     rel_l2 = paddle.norm(output_dict - output_dict) / paddle.norm(output_dict)
        ...     return {"l2": rel_l2}
        >>> metric_dict = ppsci.metric.FunctionalMetric(metric_expr)
    """

    def __init__(
        self,
        metric_expr: Optional[Callable] = None,
    ):
        super().__init__()
        self.metric_expr = metric_expr

    def forward(self, output_dict, label_dict=None, weight_dict=None):
        return self.metric_expr(output_dict)

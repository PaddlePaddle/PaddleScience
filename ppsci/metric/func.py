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

from ppsci.metric import base


class FunctionalMetric(base.Metric):
    r"""Functional metric class, which allows to use custom metric computing function from given metric_expr for complex computation cases.

    Args:
        metric_expr (Callable): expression of metric calculation.
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> def metric_expr(output_dict):
        ...     rel_l2 = 0
        ...     for key in output_dict:
        ...         length = int(len(output_dict[key])/2)
        ...         out_dict = {key: output_dict[key][:length]}
        ...         label_dict = {key: output_dict[key][length:]}
        ...         rel_l2 += paddle.norm(out_dict - label_dict) / paddle.norm(label_dict)
        ...     return {"l2": rel_l2}
        >>> metric_dict = ppsci.metric.FunctionalMetric(metric_expr)
    """

    def __init__(
        self,
        metric_expr: Callable,
        keep_batch: bool = False,
    ):
        super().__init__(keep_batch)
        self.metric_expr = metric_expr

    def forward(self, output_dict, label_dict=None):
        return self.metric_expr(output_dict)

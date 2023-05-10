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

import paddle
import paddle.nn.functional as F
from pandas import concat
from typing_extensions import Literal

from ppsci.loss import base


class L2Loss(base.Loss):
    r"""Class for l2 loss.

    $$
    L = \sum\limits_{i=1}^{N}{(x_i-y_i)^2}
    $$

    Examples:
        >>> import ppsci
        >>> loss = ppsci.loss.L2Loss()
    """

    def __init__(self):
        super().__init__()
        self.reduction = "sum"

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0.0
        for key in label_dict:
            loss = F.mse_loss(output_dict[key], label_dict[key], "none")
            if weight_dict is not None:
                loss *= weight_dict[key]
            if "area" in output_dict:
                loss *= output_dict["area"]

            loss = loss.sum()
            losses += loss
        return losses


class PeriodicL2Loss(base.Loss):
    """Class for Periodic l2 loss."""

    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        self.reduction = reduction

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0.0
        for key in label_dict:
            n_output = len(output_dict[key])
            if n_output % 2 > 0:
                raise ValueError(
                    f"Length of output({n_output}) of key({key}) should be even."
                )

            n_output //= 2
            loss = F.mse_loss(
                output_dict[key][:n_output], output_dict[key][n_output:], "none"
            )
            if weight_dict is not None:
                loss *= weight_dict[key]
            if "area" in output_dict:
                loss *= output_dict["area"]

            loss = loss.sum()
            losses += loss
        return losses


class L2RelLoss(base.Loss):
    r"""Class for l2 relative loss.

    $$
    L =
    \begin{cases}
        \dfrac{1}{N}\sum\limits_{i=1}^{N}{\dfrac{\Vert x_i-y_i\Vert_2}{\Vert y_i\Vert_2}}, & \text{if reduction='mean'} \\
        \sum\limits_{i=1}^{N}{\dfrac{\Vert x_i-y_i\Vert_2}{\Vert y_i\Vert_2}}, & \text{if reduction='sum'}
    \end{cases}
    $$

    Args:
        reduction (Literal["mean", "sum"], optional): Specifies the reduction to apply to the output: 'mean' | 'sum'. Defaults to "mean".
    
    Examples:
        >>> import ppsci
        >>> loss = ppsci.loss.L2RelLoss()
    """

    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.reduction = reduction

    def rel_loss(self, x, y):
        batch_size = x.shape[0]
        x_ = x.reshape((batch_size, -1))
        y_ = y.reshape((batch_size, -1))
        diff_norms = paddle.norm(x_ - y_, p=2, axis=1)
        y_norms = paddle.norm(y_, p=2, axis=1)
        return diff_norms / y_norms

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0
        for key in label_dict:
            loss = self.rel_loss(output_dict[key], label_dict[key])
            if weight_dict is not None:
                loss *= weight_dict[key]
            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
            losses += loss

        return losses

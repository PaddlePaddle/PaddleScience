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


import paddle.nn.functional as F

from ppsci.loss import base


class MSELoss(base.LossBase):
    """Class for mean squared error loss.

    $$
    L =
    \begin{cases}
        \frac{1}{N}\sum_{i=1}^{N}{(x_i-y_i)^2}, & \text{if reduction='mean'} \\
        \sum_{i=1}^{N}{(x_i-y_i)^2}, & \text{if reduction='sum'}
    \end{cases}
    $$

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        self.reduction = reduction

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0.0
        use_const_weight = True
        for key in label_dict:
            loss = F.mse_loss(output_dict[key], label_dict[key], "none")
            if weight_dict is not None and use_const_weight is False:
                loss *= weight_dict[key]
            if "area" in output_dict:
                loss *= output_dict["area"]

            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                if use_const_weight is True:
                    loss = loss.mean() * weight_dict[key][0]
                else:
                    loss = loss.mean()
            losses += loss
        return losses


class PeriodicMSELoss(base.LossBase):
    """Class for periodic mean squared error loss.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

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

            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
            losses += loss
        return losses

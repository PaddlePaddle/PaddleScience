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


from typing import Dict
from typing import Optional
from typing import Union

import paddle.nn.functional as F
from typing_extensions import Literal

from ppsci.loss import base


class MSELoss(base.Loss):
    r"""Class for mean squared error loss.

    $$
    L =
    \begin{cases}
        \dfrac{1}{N} \Vert {\mathbf{x}-\mathbf{y}} \Vert_2^2, & \text{if reduction='mean'} \\
        \Vert {\mathbf{x}-\mathbf{y}} \Vert_2^2, & \text{if reduction='sum'}
    \end{cases}
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import ppsci
        >>> loss = ppsci.loss.MSELoss("mean")
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        super().__init__(reduction, weight)

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0.0
        for key in label_dict:
            loss = F.mse_loss(output_dict[key], label_dict[key], "none")
            if weight_dict:
                loss *= weight_dict[key]

            if "area" in output_dict:
                loss *= output_dict["area"]

            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
            if isinstance(self.weight, (float, int)):
                loss *= self.weight
            elif isinstance(self.weight, dict) and key in self.weight:
                loss *= self.weight[key]

            losses += loss
        return losses


class MSELossWithL2Decay(MSELoss):
    r"""MSELoss with L2 decay.

    $$
    L =
    \begin{cases}
        \dfrac{1}{N} \Vert {\mathbf{x}-\mathbf{y}} \Vert_2^2 + \displaystyle\sum_{i=1}^{M}{\Vert \mathbf{K_i} \Vert_F^2}, & \text{if reduction='mean'} \\
         \Vert {\mathbf{x}-\mathbf{y}} \Vert_2^2 + \displaystyle\sum_{i=1}^{M}{\Vert \mathbf{K_i} \Vert_F^2}, & \text{if reduction='sum'}
    \end{cases}
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}, \mathbf{K_i} \in \mathcal{R}^{O_i \times P_i}
    $$

    $M$ is the number of  which apply regularization on.

    Args:
        reduction (Literal["mean", "sum"], optional): Specifies the reduction to apply to the output: 'mean' | 'sum'. Defaults to "mean".
        regularization_dict (Optional[Dict[str, float]]): Regularization dictionary. Defaults to None.
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Raises:
        ValueError: reduction should be 'mean' or 'sum'.

    Examples:
        >>> import ppsci
        >>> loss = ppsci.loss.MSELossWithL2Decay("mean", {"k_matrix": 2.0})
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        regularization_dict: Optional[Dict[str, float]] = None,
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        super().__init__(reduction, weight)
        self.regularization_dict = regularization_dict

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = super().forward(output_dict, label_dict, weight_dict)

        if self.regularization_dict is not None:
            for reg_key, reg_weight in self.regularization_dict.items():
                loss = output_dict[reg_key].pow(2).sum()
                losses += loss * reg_weight
        return losses


class PeriodicMSELoss(base.Loss):
    r"""Class for periodic mean squared error loss.

    $$
    L =
    \begin{cases}
        \dfrac{1}{N} \Vert \mathbf{x_l}-\mathbf{x_r} \Vert_2^2, & \text{if reduction='mean'} \\
        \Vert \mathbf{x_l}-\mathbf{x_r} \Vert_2^2, & \text{if reduction='sum'}
    \end{cases}
    $$

    $\mathbf{x_l} \in \mathcal{R}^{N}$ is the first half of batch output,
    $\mathbf{x_r} \in \mathcal{R}^{N}$ is the second half of batch output.

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        super().__init__(reduction, weight)

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

            if isinstance(self.weight, (float, int)):
                loss *= self.weight
            elif isinstance(self.weight, dict) and key in self.weight:
                loss *= self.weight[key]

            losses += loss
        return losses

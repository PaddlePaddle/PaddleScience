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

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Union

import paddle.nn.functional as F
from typing_extensions import Literal

from ppsci.loss import base


class IntegralLoss(base.Loss):
    r"""Class for integral loss with Monte-Carlo integration algorithm.

    $$
    L =
    \begin{cases}
        \dfrac{1}{N} \Vert \displaystyle\sum_{i=1}^{M}{\mathbf{s}_i \cdot \mathbf{x}_i} - \mathbf{y} \Vert_2^2, & \text{if reduction='mean'} \\
         \Vert \displaystyle\sum_{i=0}^{M}{\mathbf{s}_i \cdot \mathbf{x}_i} - \mathbf{y} \Vert_2^2, & \text{if reduction='sum'}
    \end{cases}
    $$

    $$
    \mathbf{x}, \mathbf{s} \in \mathcal{R}^{M \times N}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import IntegralLoss

        >>> output_dict = {'u': paddle.to_tensor([[0.5, 2.2, 0.9], [1.1, 0.8, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 2.2, 0.9], [1.1, 0.8, -1.3]]),
        ...                'area': paddle.to_tensor([[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]])}
        >>> label_dict = {'u': paddle.to_tensor([-1.8, 0.0]),
        ...               'v': paddle.to_tensor([0.1, 0.1])}
        >>> weight = {'u': 0.8, 'v': 0.2}
        >>> loss = IntegralLoss(weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               1.40911996)

        >>> loss = IntegralLoss(reduction="sum", weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               2.81823993)
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
            loss = F.mse_loss(
                (output_dict[key] * output_dict["area"]).sum(axis=1),
                label_dict[key],
                "none",
            )
            if weight_dict and key in weight_dict:
                loss *= weight_dict[key]

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

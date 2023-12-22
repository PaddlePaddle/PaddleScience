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


class L1Loss(base.Loss):
    r"""Class for l1 loss.

    $$
    L = \Vert \mathbf{x} - \mathbf{y} \Vert_1
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    when `reduction` is set to "mean"

    $$
    L = MEAN \left( \Vert \mathbf{x} - \mathbf{y} \Vert_1 \right)
    $$

    when `reduction` is set to "sum"

    $$
    L = SUM \left( \Vert \mathbf{x} - \mathbf{y} \Vert_1 \right)
    $$

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import L1Loss
        >>> output_dict = {"u": paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...                "v": paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {"u": paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...               "v": paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> weight = {"u": 0.8, "v": 0.2}
        >>> loss = L1Loss(weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               3.35999990)

        >>> loss = L1Loss(reduction="sum", weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               6.71999979)
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
            loss = F.l1_loss(output_dict[key], label_dict[key], "none")
            if weight_dict and key in weight_dict:
                loss *= weight_dict[key]

            if "area" in output_dict:
                loss *= output_dict["area"]

            loss = loss.sum(axis=1)

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


class PeriodicL1Loss(base.Loss):
    r"""Class for periodic l1 loss.

    $$
    L = \Vert \mathbf{x_l}-\mathbf{x_r} \Vert_1
    $$

    $\mathbf{x_l} \in \mathcal{R}^{N}$ is the first half of batch output,
    $\mathbf{x_r} \in \mathcal{R}^{N}$ is the second half of batch output.

    when `reduction` is set to "mean"

    $$
    L = MEAN \left( \Vert \mathbf{x_l}-\mathbf{x_r} \Vert_1 \right)
    $$

    when `reduction` is set to "sum"

    $$
    L = SUM \left( \Vert \mathbf{x_l}-\mathbf{x_r} \Vert_1 \right)
    $$

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import PeriodicL1Loss

        >>> output_dict = {'u': paddle.to_tensor([[0.5, 2.2, 0.9], [1.1, 0.8, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 2.2, 0.9], [1.1, 0.8, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 0.0, 1.0], [-0.2, 0.2, 2.5]]),
        ...               'v': paddle.to_tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])}
        >>> weight = {'u': 0.8, 'v': 0.2}
        >>> loss = PeriodicL1Loss(weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               4.19999981)

        >>> loss = PeriodicL1Loss(reduction="sum", weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               4.19999981)
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
            loss = F.l1_loss(
                output_dict[key][:n_output], output_dict[key][n_output:], "none"
            )
            if weight_dict and key in weight_dict:
                loss *= weight_dict[key]
            if "area" in output_dict:
                loss *= output_dict["area"]

            loss = loss.sum(axis=1)

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

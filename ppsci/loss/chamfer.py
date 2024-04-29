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

import paddle

from ppsci.loss import base


class ChamferLoss(base.Loss):
    r"""Class for Chamfe distance loss.

    $$
    L = \dfrac{1}{S_1} \sum_{x \in S_1} \min_{y \in S_2} \Vert x - y \Vert_2^2 + \dfrac{1}{S_2} \sum_{y \in S_2} \min_{x \in S_1} \Vert y - x \Vert_2^2
    $$

    $$
    \text{where } S_1 \text{ and } S_2 \text{ is the coordinate matrix of two point clouds}.
    $$

    Args:
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import ChamferLoss
        >>> _ = paddle.seed(42)
        >>> batch_point_cloud1 = paddle.rand([2, 100, 3])
        >>> batch_point_cloud2 = paddle.rand([2, 50, 3])
        >>> output_dict = {"s1": batch_point_cloud1}
        >>> label_dict  = {"s1": batch_point_cloud2}
        >>> weight = {"s1": 0.8}
        >>> loss = ChamferLoss(weight=weight)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               0.04415882)
    """

    def __init__(
        self,
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        super().__init__("mean", weight)

    def forward(self, output_dict, label_dict, weight_dict=None):
        losses = 0.0
        for key in label_dict:
            s1 = output_dict[key]
            s2 = label_dict[key]
            N1, N2 = s1.shape[1], s2.shape[1]

            # [B, N1, N2, 3]
            s1_expand = paddle.expand(s1.reshape([-1, N1, 1, 3]), shape=[-1, N1, N2, 3])
            # [B, N1, N2, 3]
            s2_expand = paddle.expand(s2.reshape([-1, 1, N2, 3]), shape=[-1, N1, N2, 3])

            dis = ((s1_expand - s2_expand) ** 2).sum(axis=3)  # [B, N1, N2]
            loss_s12 = dis.min(axis=2)  # [B, N1]
            loss_s21 = dis.min(axis=1)  # [B, N2]
            loss = loss_s12.mean() + loss_s21.mean()

            if weight_dict and key in weight_dict:
                loss *= weight_dict[key]

            if isinstance(self.weight, (float, int)):
                loss *= self.weight
            elif isinstance(self.weight, dict) and key in self.weight:
                loss *= self.weight[key]

            losses += loss
        return losses

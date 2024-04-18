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

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import paddle

from ppsci.loss import base


class FunctionalLoss(base.Loss):
    r"""Functional loss class, which allows to use custom loss computing function from given loss_expr for complex computation cases.

    $$
    L = f(\mathbf{x}, \mathbf{y})
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        loss_expr (Callable[..., paddle.Tensor]): Function for custom loss computation.
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import FunctionalLoss
        >>> import paddle.nn.functional as F
        >>> def mse_sum_loss(output_dict, label_dict, weight_dict=None):
        ...     losses = 0
        ...     for key in output_dict.keys():
        ...         loss = F.mse_loss(output_dict[key], label_dict[key], "sum")
        ...         if weight_dict:
        ...             loss *=  weight_dict[key]
        ...         losses += loss
        ...     return losses
        >>> loss = FunctionalLoss(mse_sum_loss)
        >>> output_dict = {'u': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...             'v': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...             'v': paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> weight_dict = {'u': 0.8, 'v': 0.2}
        >>> result = loss(output_dict, label_dict, weight_dict)
        >>> print(result)
        Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               17.89600182)
    """

    def __init__(
        self,
        loss_expr: Callable[..., paddle.Tensor],
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        super().__init__(None, weight)
        self.loss_expr = loss_expr

    def forward(self, output_dict, label_dict=None, weight_dict=None) -> paddle.Tensor:
        loss = self.loss_expr(output_dict, label_dict, weight_dict)

        assert isinstance(
            loss, (paddle.Tensor, paddle.static.Variable, paddle.pir.Value)
        ), (
            "Loss computed by custom function should be type of 'paddle.Tensor', "
            f"'paddle.static.Variable' or 'paddle.pir.Value', but got {type(loss)}."
            " Please check the return type of custom loss function."
        )

        return loss

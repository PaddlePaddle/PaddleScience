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
from typing import Dict
from typing import Optional
from typing import Union

from typing_extensions import Literal

from ppsci.loss import base


class FunctionalLoss(base.Loss):
    r"""Functional loss class, which allows to use custom loss computing function from given loss_expr for complex computation cases.

    Args:
        loss_expr (Callable): expression of loss calculation.
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Weight for loss. Defaults to None.

    Examples:
        >>> import ppsci
        >>> import paddle.nn.functional as F
        >>> def loss_expr(output_dict):
        ...     losses = 0
        ...     for key in output_dict:
        ...         length = int(len(output_dict[key])/2)
        ...         out_dict = {key: output_dict[key][:length]}
        ...         label_dict = {key: output_dict[key][length:]}
        ...         losses += F.mse_loss(out_dict, label_dict, "sum")
        ...     return losses
        >>> loss = ppsci.loss.FunctionalLoss(loss_expr)
    """

    def __init__(
        self,
        loss_expr: Callable,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        super().__init__(reduction, weight)
        self.loss_expr = loss_expr

    def forward(self, output_dict, label_dict=None, weight_dict=None):
        return self.loss_expr(output_dict)

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

from ppsci.loss import base


class FunctionalLoss(base.Loss):
    r"""Class for functional loss.

    Args:
        loss_expr (Optional[Callable], optional): expression of loss calculation. Defaults to None.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> def loss_expr(output_dict):
        ...     return paddle.nn.functional.mse_loss(output_dict, output_dict, "sum")
        >>> loss = ppsci.loss.FunctionalLoss(loss_expr)
    """

    def __init__(
        self,
        loss_expr: Optional[Callable] = None,
    ):
        super().__init__("mean", None)
        self.loss_expr = loss_expr

    def forward(self, output_dict, label_dict=None, weight_dict=None):
        return self.loss_expr(output_dict)

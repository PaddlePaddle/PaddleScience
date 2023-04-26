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

from ppsci.loss import base


class IntegralLoss(base.LossBase):
    """Class for integral loss.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[Dict[str, float], float]]): Weight for loss. Defaults to None.
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[Union[Dict[str, float], float]] = None,
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
            if weight_dict is not None:
                loss *= weight_dict[key]
            if isinstance(self.weight, float):
                loss *= self.weight
            elif isinstance(self.weight, dict) and key in self.weight:
                loss *= self.weight[key]

            if self.reduction == "sum":
                loss = loss.sum()
            elif self.reduction == "mean":
                loss = loss.mean()
            losses += loss
        return losses

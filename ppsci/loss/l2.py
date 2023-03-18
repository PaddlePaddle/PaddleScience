"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import paddle.nn.functional as F

from ppsci.loss import base


class L2Loss(base.LossBase):
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

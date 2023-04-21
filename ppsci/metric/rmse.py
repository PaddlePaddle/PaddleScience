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

import paddle
import paddle.nn.functional as F

from ppsci.metric import base


class RMSE(base.MetricBase):
    r"""Root mean square error

    $$
    metric = \sqrt{\dfrac{1}{N}\sum\limits_{i=1}^{N}{(x_i-y_i)^2}}
    $$
    """

    def __init__(self):
        super().__init__()

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in output_dict:
            rmse = F.mse_loss(output_dict[key], label_dict[key], "mean") ** 0.5
            metric_dict[key] = float(rmse)

        return metric_dict

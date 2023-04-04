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
import paddle.nn as nn


class L2Rel(nn.Layer):
    r"""Class for l2 relative error.

    $$
    metric = \frac{\Vert x-y \Vert_2}{\Vert y \Vert_2}
    $$
    """

    def __init__(self):
        super().__init__()

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in output_dict:
            rel_l2 = paddle.norm(label_dict[key] - output_dict[key]) / paddle.norm(
                label_dict[key]
            )
            metric_dict[key] = float(rel_l2)

        return metric_dict

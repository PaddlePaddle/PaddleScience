# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn


# loss function with rel/abs Lp loss
class LpLoss(nn.Layer):
    def __init__(self, batch_size=20, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.batch_size = batch_size

    def abs(self, x, y):
        num_examples = x.shape[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * paddle.norm(
            x.reshape([num_examples, -1]) - y.reshape([num_examples, -1]), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return (all_norms).mean()
            else:
                return (all_norms).sum()

        return all_norms

    def rel(self, output_dict, label_dict, weight_dict=None):
        x = output_dict["output"]
        y = label_dict["output"]
        x = x.view([self.batch_size, -1])
        y = y.view([self.batch_size, -1])
        num_examples = x.shape[0]

        diff_norms = paddle.norm(
            x.reshape([num_examples, -1]) - y.reshape([num_examples, -1]), self.p, 1
        )
        y_norms = paddle.norm(y.reshape([num_examples, -1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return (diff_norms / y_norms).mean()
            else:
                return (diff_norms / y_norms).sum()

        return diff_norms / y_norms

    def forward(self, x, y):
        return {"output": self.rel(x, y)}

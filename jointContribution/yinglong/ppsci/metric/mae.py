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


class MAE(base.Metric):
    r"""Mean absolute error.

    $$
    metric = \dfrac{1}{N} \Vert \mathbf{x} - \mathbf{y} \Vert_1
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import ppsci
        >>> metric = ppsci.metric.MAE()
    """

    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            mae = F.l1_loss(output_dict[key], label_dict[key], "none")
            if self.keep_batch:
                metric_dict[key] = mae.mean(axis=tuple(range(1, mae.ndim)))
            else:
                metric_dict[key] = mae.mean()

        return metric_dict

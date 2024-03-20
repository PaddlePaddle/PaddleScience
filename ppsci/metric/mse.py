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

import paddle
import paddle.nn.functional as F

from ppsci.metric import base


class MSE(base.Metric):
    r"""Mean square error

    $$
    metric = \dfrac{1}{N} \Vert \mathbf{x} - \mathbf{y} \Vert_2^2
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import paddle
        >>> from ppsci.metric import MSE
        >>> output_dict = {'u': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...               'v': paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> loss = MSE()
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               5.35750008), 'v': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               0.94000000)}
        >>> loss = MSE(keep_batch=True)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [2.65000010, 8.06499958]), 'v': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [0.39999998, 1.48000002])}
    """

    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            mse = F.mse_loss(output_dict[key], label_dict[key], "none")
            if self.keep_batch:
                metric_dict[key] = mse.mean(axis=tuple(range(1, mse.ndim)))
            else:
                metric_dict[key] = mse.mean()

        return metric_dict

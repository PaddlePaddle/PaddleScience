# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

from typing import Dict

import paddle

from ppsci.metric import base


class MaxAE(base.Metric):
    r"""Maximum Absolute Error (MaxAE).

    $$
    \text{MaxAE} = \max_i \left( |x_i - y_i| \right)
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import paddle
        >>> from ppsci.metric import MaxAE
        >>> output_dict = {'u': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...               'v': paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> metric = MaxAE()
        >>> result = metric(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               3.80000007), 'v': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               0.80000001)}
        >>> metric = MaxAE(keep_batch=True)
        >>> result = metric(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [1.30000002, 3.80000007]), 'v': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [0.40000001, 0.80000001])}
    """

    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict) -> Dict[str, "paddle.Tensor"]:
        maxae_dict = {}

        for key in label_dict:
            # Calculate absolute error
            ae = paddle.abs(output_dict[key] - label_dict[key])

            if self.keep_batch:
                # Take the maximum AE within each batch
                maxae_dict[key] = paddle.amax(ae, axis=tuple(range(1, ae.ndim)))
            else:
                # Take the global maximum AE across all elements
                maxae_dict[key] = paddle.amax(ae)

        return maxae_dict

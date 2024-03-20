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

import numpy as np
import paddle

from ppsci.metric import base


class L2Rel(base.Metric):
    r"""Class for l2 relative error.

    NOTE: This metric API is slightly different from `MeanL2Rel`, difference is as below:

    - `L2Rel` regards the input sample as a whole and calculates the l2 relative error of the whole;
    - `MeanL2Rel` will calculate L2Rel separately for each input sample and return the average of l2 relative error for all samples.

    $$
    metric = \dfrac{\Vert \mathbf{x} - \mathbf{y} \Vert_2}{\max(\Vert \mathbf{y} \Vert_2, \epsilon)}
    $$

    $$
    \mathbf{x}, \mathbf{y} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import paddle
        >>> from ppsci.metric import L2Rel
        >>> output_dict = {'u': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...               'v': paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> loss = L2Rel()
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               1.42658269), 'v': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               9.69535923)}
    """

    # NOTE: Avoid divide by zero in result
    # see https://github.com/scikit-learn/scikit-learn/pull/15007
    EPS: float = np.finfo(np.float32).eps

    def __init__(self, keep_batch: bool = False):
        if keep_batch:
            raise ValueError(f"keep_batch should be False, but got {keep_batch}.")
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            rel_l2 = paddle.norm(label_dict[key] - output_dict[key], p=2) / paddle.norm(
                label_dict[key], p=2
            ).clip(min=self.EPS)
            metric_dict[key] = rel_l2

        return metric_dict


class MeanL2Rel(base.Metric):
    r"""Class for mean l2 relative error.

    NOTE: This metric API is slightly different from `L2Rel`, difference is as below:

    - `MeanL2Rel` will calculate L2Rel separately for each input sample and return the average of l2 relative error for all samples.
    - `L2Rel` regards the input sample as a whole and calculates the l2 relative error of the whole;

    $$
    metric = \dfrac{1}{M} \sum_{i=1}^{M}\dfrac{\Vert \mathbf{x_i} - \mathbf{y_i} \Vert_2}{\max(\Vert \mathbf{y_i} \Vert_2, \epsilon) }
    $$

    $$
    \mathbf{x_i}, \mathbf{y_i} \in \mathcal{R}^{N}
    $$

    Args:
        keep_batch (bool, optional): Whether keep batch axis. Defaults to False.

    Examples:
        >>> import paddle
        >>> from ppsci.metric import MeanL2Rel
        >>> output_dict = {'u': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        ...                'v': paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]])}
        >>> label_dict = {'u': paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        ...               'v': paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]])}
        >>> loss = MeanL2Rel()
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               1.35970235), 'v': Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               9.24504089)}
        >>> loss = MeanL2Rel(keep_batch=True)
        >>> result = loss(output_dict, label_dict)
        >>> print(result)
        {'u': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [1.11803389, 1.60137081]), 'v': Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
               [6.32455540 , 12.16552544])}
    """

    # NOTE: Avoid divide by zero in result
    # see https://github.com/scikit-learn/scikit-learn/pull/15007
    EPS: float = np.finfo(np.float32).eps

    def __init__(self, keep_batch: bool = False):
        super().__init__(keep_batch)

    @paddle.no_grad()
    def forward(self, output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            rel_l2 = paddle.norm(
                label_dict[key] - output_dict[key], p=2, axis=1
            ) / paddle.norm(label_dict[key], p=2, axis=1).clip(min=self.EPS)
            if self.keep_batch:
                metric_dict[key] = rel_l2
            else:
                metric_dict[key] = rel_l2.mean()

        return metric_dict

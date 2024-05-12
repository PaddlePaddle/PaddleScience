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

from typing import List

import numpy as np
import paddle
from paddle import nn

from ppsci.loss.mtl import base


class PCGrad(base.LossAggregator):
    r"""
    **P**rojecting **C**onflicting Gradients

    [Gradient Surgery for Multi-Task Learning](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

    Code reference: [https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py)

    Args:
        model (nn.Layer): Training model.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import mtl
        >>> model = paddle.nn.Linear(3, 4)
        >>> loss_aggregator = mtl.PCGrad(model)
        >>> for i in range(5):
        ...     x1 = paddle.randn([8, 3])
        ...     x2 = paddle.randn([8, 3])
        ...     y1 = model(x1)
        ...     y2 = model(x2)
        ...     loss1 = paddle.sum(y1)
        ...     loss2 = paddle.sum((y2 - 2) ** 2)
        ...     loss_aggregator([loss1, loss2]).backward()
    """

    def __init__(self, model: nn.Layer) -> None:
        super().__init__(model)
        self._zero = paddle.zeros([])

    def backward(self) -> None:
        np.random.shuffle(self.losses)
        grads_list = self._compute_grads()
        with paddle.no_grad():
            refined_grads = self._refine_grads(grads_list)
            self._set_grads(refined_grads)

    def _compute_grads(self) -> List[paddle.Tensor]:
        # compute all gradients derived by each loss
        grads_list = []  # num_params x num_losses
        for loss in self.losses:
            # backward with current loss
            loss.backward()
            grads_list.append(
                paddle.concat(
                    [
                        param.grad.clone().reshape([-1])
                        for param in self.model.parameters()
                        if param.grad is not None
                    ],
                    axis=0,
                )
            )
            # clear gradients for current loss for not affecting other loss
            self.model.clear_gradients()

        return grads_list

    def _refine_grads(self, grads_list: List[paddle.Tensor]) -> List[paddle.Tensor]:
        def proj_grad(grad: paddle.Tensor):
            for k in range(self.loss_num):
                inner_product = paddle.sum(grad * grads_list[k])
                proj_direction = inner_product / paddle.sum(
                    grads_list[k] * grads_list[k]
                )
                grad = grad - paddle.minimum(proj_direction, self._zero) * grads_list[k]
            return grad

        grads_list = [proj_grad(grad) for grad in grads_list]

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads: List[paddle.Tensor] = []
        for j in range(self.loss_num):
            start_idx = 0
            for idx, var in enumerate(self.model.parameters()):
                grad_shape = var.shape
                flatten_dim = var.numel()
                refined_grad = grads_list[j][start_idx : start_idx + flatten_dim]
                refined_grad = paddle.reshape(refined_grad, grad_shape)
                if len(proj_grads) < self.param_num:
                    proj_grads.append(refined_grad)
                else:
                    proj_grads[idx] += refined_grad
                start_idx += flatten_dim
        return proj_grads

    def _set_grads(self, grads_list: List[paddle.Tensor]) -> None:
        for i, param in enumerate(self.model.parameters()):
            param.grad = grads_list[i]

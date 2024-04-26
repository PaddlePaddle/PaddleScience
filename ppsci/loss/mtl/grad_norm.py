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

import paddle
from paddle import nn

from ppsci.loss.mtl import base


class GradNorm(base.LossAggregator):
    r"""GradNorm loss weighting algorithm.

    reference: [https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/models.py#L132-L146](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/models.py#L132-L146)

    $$
    \begin{align*}
    L^t &= \sum_{i=1}^{N}{\tilde{w}_i^t\cdot L_i^t}, \\
        \text{where } \\
        \tilde{w}_i^0&=1, \\
        \tilde{w}_i^t&=\tilde{w}_i^{t-1}\cdot m+w_i^t\cdot (1-m), t\ge1\\
        w_i^t&=\dfrac{\overline{\Vert \nabla_{\theta}{L_i^t} \Vert_2}}{\Vert \nabla_{\theta}{L_i^t} \Vert_2}, \\
        \overline{\Vert \nabla_{\theta}{L_i^t} \Vert_2}&=\dfrac{1}{N}\sum_{i=1}^N{\Vert \nabla_{\theta}{L_i^t} \Vert_2}, \\
        &t \text{ is the training step started from 0}.
    \end{align*}
    $$

    Args:
        model (nn.Layer): Training model.
        num_losses (int, optional): Number of losses. Defaults to 1.
        update_freq (int, optional): Weight updating frequency. Defaults to 1000.
        momentum (float, optional): Momentum $m$ for moving weight. Defaults to 0.9.
    """

    def __init__(
        self,
        model: nn.Layer,
        num_losses: int = 1,
        update_freq: int = 1000,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(model)
        self.step = 0
        self.num_losses = num_losses
        self.update_freq = update_freq
        self.momentum = momentum
        self.register_buffer("weight", paddle.ones([num_losses]))

    def _compute_weight(self, losses: List["paddle.Tensor"]) -> List["paddle.Tensor"]:
        grad_norms = []
        for loss in losses:
            loss.backward(retain_graph=True)  # NOTE: Keep graph for loss backward
            with paddle.no_grad():
                grad_vector = paddle.concat(
                    [
                        p.grad.reshape([-1])
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                )
                grad_norms.append(paddle.linalg.norm(grad_vector, p=2))
                self.model.clear_gradients()

        mean_grad_norm = paddle.mean(paddle.stack(grad_norms))
        weight = [(mean_grad_norm / x) for x in grad_norms]

        return weight

    def __call__(self, losses: List["paddle.Tensor"], step: int = 0) -> "paddle.Tensor":
        assert len(losses) == self.num_losses, (
            f"Length of given losses({len(losses)}) should be equal to "
            f"num_losses({self.num_losses})."
        )
        self.step = step

        # compute current loss with moving weights
        loss = self.weight[0] * losses[0]
        for i in range(1, len(losses)):
            loss += self.weight[i] * losses[i]

        # update moving weights every 'update_freq' steps
        if self.step % self.update_freq == 0:
            weight = self._compute_weight(losses)
            for i in range(self.num_losses):
                self.weight[i].set_value(
                    self.momentum * self.weight[i] + (1 - self.momentum) * weight[i]
                )

        return loss

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


class Relobralo(nn.Layer):
    """Base class of loss aggregator mainly for multitask learning.

    Args:
        model (nn.Layer): Training model.
    """

    def __init__(
        self,
        model: nn.Layer,
        num_losses: int,
        alpha: float = 0.95,
        beta: float = 0.99,
        tau: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.model = model
        self.step = 0
        self.param_num = 0
        for param in self.model.parameters():
            if not param.stop_gradient:
                self.param_num += 1
        self.num_losses: int = num_losses
        self.alpha: float = alpha
        self.beta: float = beta
        self.tau: float = tau
        self.eps: float = eps
        self.register_buffer("losses_init", paddle.zeros([self.num_losses]))
        self.register_buffer("losses_prev", paddle.zeros([self.num_losses]))
        self.register_buffer("lmbda", paddle.ones([self.num_losses]))

    def _softmax(self, vec: paddle.Tensor) -> paddle.Tensor:
        max_item = vec.max()
        result = paddle.exp(vec - max_item) / paddle.exp(vec - max_item).sum()
        return result

    def _compute_bal(
        self, losses_vec1: paddle.Tensor, losses_vec2: paddle.Tensor
    ) -> paddle.Tensor:
        return self.num_losses * (
            self._softmax(losses_vec1 / (self.tau * losses_vec2 + self.eps))
        )

    def __call__(self, losses: List[paddle.Tensor], step: int = 0) -> "Relobralo":
        self.step = step
        losses_stacked = paddle.stack(losses)  # [num_losses, ]

        if self.step == 0:
            self.loss = losses_stacked.sum()
            with paddle.no_grad():
                paddle.assign(losses_stacked.detach(), self.losses_init)
        else:
            with paddle.no_grad():
                # 1. update lambda_hist
                rho = paddle.bernoulli(paddle.to_tensor(self.beta))
                lmbda_hist = rho * self.lmbda + (1 - rho) * self._compute_bal(
                    losses_stacked, self.losses_init
                )

                # 2. update lambda
                paddle.assign(
                    self.alpha * lmbda_hist
                    + (1 - self.alpha)
                    * self._compute_bal(losses_stacked, self.losses_prev),
                    self.lmbda,
                )

            # 3. compute reweighted total loss with lambda
            self.loss = (losses_stacked * self.lmbda).sum()

        # update losses_prev at the end of each step
        with paddle.no_grad():
            paddle.assign(losses_stacked.detach(), self.losses_prev)
        return self

    def backward(self) -> None:
        self.loss.backward()

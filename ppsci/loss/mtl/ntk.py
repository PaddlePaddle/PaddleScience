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


class NTK(base.LossAggregator):
    def __init__(
        self,
        model: nn.Layer,
        num_losses: int = 1,
        update_freq: int = 1000,
    ) -> None:
        super().__init__(model)
        self.step = 0
        self.num_losses = num_losses
        self.update_freq = update_freq
        self.register_buffer("weight", paddle.ones([num_losses]))

    def _compute_weight(self, losses):
        ntk_sum = 0
        ntk_value = []
        for loss in losses:
            loss.backward(retain_graph=True)  # NOTE: Keep graph for loss backward
            with paddle.no_grad():
                grad = paddle.concat(
                    [
                        p.grad.reshape([-1])
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                )
                ntk_value.append(
                    paddle.sqrt(
                        paddle.sum(grad.detach() ** 2),
                    )
                )

        ntk_sum += paddle.sum(paddle.stack(ntk_value, axis=0))
        ntk_weight = [(ntk_sum / x) for x in ntk_value]

        return ntk_weight

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
            computed_weight = self._compute_weight(losses)
            for i in range(self.num_losses):
                self.weight[i].set_value(computed_weight[i])

        return loss

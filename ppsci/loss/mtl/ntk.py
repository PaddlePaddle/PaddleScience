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

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Dict

import paddle

from ppsci.loss.mtl import base

if TYPE_CHECKING:
    from paddle import nn


class NTK(base.LossAggregator):
    r"""Weighted Neural Tangent Kernel.

    reference: [https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/models.py#L148-L158](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/models.py#L148-L158)

    Attributes:
        should_persist (bool): Whether to persist the loss aggregator when saving.
            Those loss aggregators with parameters and/or buffers should be persisted.

    Args:
        model (nn.Layer): Training model.
        num_losses (int, optional): Number of losses. Defaults to 1.
        update_freq (int, optional): Weight updating frequency. Defaults to 1000.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import mtl
        >>> model = paddle.nn.Linear(3, 4)
        >>> loss_aggregator = mtl.NTK(model, num_losses=2)
        >>> for i in range(5):
        ...     x1 = paddle.randn([8, 3])
        ...     x2 = paddle.randn([8, 3])
        ...     y1 = model(x1)
        ...     y2 = model(x2)
        ...     loss1 = paddle.sum(y1)
        ...     loss2 = paddle.sum((y2 - 2) ** 2)
        ...     loss_aggregator({'loss1': loss1, 'loss2': loss2}).backward()
    """
    should_persist: ClassVar[bool] = True
    weight: paddle.Tensor

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

    def _compute_weight(self, losses: Dict[str, paddle.Tensor]):
        ntk_sum = 0
        ntk_value = []
        for loss in losses.values():
            grads = paddle.grad(
                loss,
                self.model.parameters(),
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
            with paddle.no_grad():
                grad = paddle.concat(
                    [grad.reshape([-1]) for grad in grads if grad is not None]
                )
                ntk_value.append(
                    paddle.sqrt(
                        paddle.sum(grad.detach() ** 2),
                    )
                )

        ntk_sum += paddle.sum(paddle.stack(ntk_value, axis=0))
        ntk_weight = [(ntk_sum / x) for x in ntk_value]

        return ntk_weight

    def __call__(
        self, losses: Dict[str, "paddle.Tensor"], step: int = 0
    ) -> "paddle.Tensor":
        assert len(losses) == self.num_losses, (
            f"Length of given losses({len(losses)}) should be equal to "
            f"num_losses({self.num_losses})."
        )
        self.step = step

        # compute current loss with moving weights
        loss = 0
        for i, (k, v) in enumerate(losses.items()):
            loss = loss + self.weight[i] * v

        # update moving weights every 'update_freq' steps
        if self.step % self.update_freq == 0:
            computed_weight = self._compute_weight(losses)
            for i in range(self.num_losses):
                self.weight[i].set_value(computed_weight[i])

        return loss

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

from typing import ClassVar
from typing import Dict
from typing import List

import paddle
from paddle import nn

from ppsci.loss.mtl import base

# from ppsci.utils import logger


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

    Attributes:
        should_persist(bool): Whether to persist the loss aggregator when saving.
            Those loss aggregators with parameters and/or buffers should be persisted.

    Args:
        model (nn.Layer): Training model.
        num_losses (int, optional): Number of losses. Defaults to 1.
        update_freq (int, optional): Weight updating frequency. Defaults to 1000.
        momentum (float, optional): Momentum $m$ for moving weight. Defaults to 0.9.
        init_weights (List[float]): Initial weights list. Defaults to None.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import mtl
        >>> model = paddle.nn.Linear(3, 4)
        >>> loss_aggregator = mtl.GradNorm(model, num_losses=2)
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
        momentum: float = 0.9,
        init_weights: List[float] = None,
    ) -> None:
        super().__init__(model)
        self.step = 0
        self.num_losses = num_losses
        self.update_freq = update_freq
        self.momentum = momentum
        if init_weights is not None and num_losses != len(init_weights):
            raise ValueError(
                f"Length of init_weights({len(init_weights)}) should be equal to "
                f"num_losses({num_losses})."
            )
        self.register_buffer(
            "weight",
            paddle.to_tensor(init_weights, dtype="float32")
            if init_weights is not None
            else paddle.ones([num_losses]),
        )

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

    def __call__(
        self, losses: Dict[str, "paddle.Tensor"], step: int = 0
    ) -> "paddle.Tensor":
        assert len(losses) == self.num_losses, (
            f"Length of given losses({len(losses)}) should be equal to "
            f"num_losses({self.num_losses})."
        )
        self.step = step

        # compute current loss with moving weights
        loss = 0.0
        for i, key in enumerate(losses):
            if i == 0:
                loss = self.weight[i] * losses[key]
            else:
                loss += self.weight[i] * losses[key]

        # update moving weights every 'update_freq' steps
        if self.step % self.update_freq == 0:
            weight = self._compute_weight(list(losses.values()))
            for i in range(self.num_losses):
                self.weight[i].set_value(
                    self.momentum * self.weight[i] + (1 - self.momentum) * weight[i]
                )
            # logger.message(f"weight at step {self.step}: {self.weight.numpy()}")

        return loss

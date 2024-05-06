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


class AGDA(base.LossAggregator):
    r"""
    **A**daptive **G**radient **D**escent **A**lgorithm

    [Physics-informed neural network based on a new adaptive gradient descent algorithm for solving partial differential equations of flow problems](https://pubs.aip.org/aip/pof/article-abstract/35/6/063608/2899773/Physics-informed-neural-network-based-on-a-new)

    Args:
        model (nn.Layer): Training model.
        M (int, optional): Smoothing period. Defaults to 100.
        gamma (float, optional): Smooth factor. Defaults to 0.999.

    Examples:
        >>> import paddle
        >>> from ppsci.loss import mtl
        >>> model = paddle.nn.Linear(3, 4)
        >>> loss_aggregator = mtl.AGDA(model)
        >>> for i in range(5):
        ...     x1 = paddle.randn([8, 3])
        ...     x2 = paddle.randn([8, 3])
        ...     y1 = model(x1)
        ...     y2 = model(x2)
        ...     loss1 = paddle.sum(y1)
        ...     loss2 = paddle.sum((y2 - 2) ** 2)
        ...     loss_aggregator([loss1, loss2]).backward()
    """

    def __init__(self, model: nn.Layer, M: int = 100, gamma: float = 0.999) -> None:
        super().__init__(model)
        self.M = M
        self.gamma = gamma
        self.Lf_smooth = 0
        self.Lu_smooth = 0
        self.Lf_tilde_acc = 0.0
        self.Lu_tilde_acc = 0.0

    def __call__(self, losses, step: int = 0) -> "AGDA":
        if len(losses) != 2:
            raise ValueError(
                f"Number of losses(tasks) for AGDA shoule be 2, but got {len(losses)}"
            )
        return super().__call__(losses, step)

    def backward(self) -> None:
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
        # compute moving average of L^smooth_i(n) - eq.(16)
        self.Lf_smooth = (
            self.gamma * self.Lf_smooth + (1 - self.gamma) * self.losses[0].item()
        )
        self.Lu_smooth = (
            self.gamma * self.Lu_smooth + (1 - self.gamma) * self.losses[1].item()
        )

        # compute L^smooth_i(kM) - eq.(17)
        if self.step % self.M == 0:
            Lf_smooth_kM = self.Lf_smooth
            Lu_smooth_kM = self.Lu_smooth
        Lf_tilde = self.Lf_smooth / Lf_smooth_kM
        Lu_tilde = self.Lu_smooth / Lu_smooth_kM

        # compute r_i(n) - eq.(18)
        self.Lf_tilde_acc += Lf_tilde
        self.Lu_tilde_acc += Lu_tilde
        rf = Lf_tilde / self.Lf_tilde_acc
        ru = Lu_tilde / self.Lu_tilde_acc

        # compute E(g(n)) - step1(1)
        gf_magn = (grads_list[0] * grads_list[0]).sum().sqrt()
        gu_magn = (grads_list[1] * grads_list[1]).sum().sqrt()
        Eg = (gf_magn + gu_magn) / 2

        # compute \omega_f(n) - step1(2)
        omega_f = (rf * (Eg - gf_magn) + gf_magn) / gf_magn
        omega_u = (ru * (Eg - gu_magn) + gu_magn) / gu_magn

        # compute g_bar(n) - step1(3)
        gf_bar = omega_f * grads_list[0]
        gu_bar = omega_u * grads_list[1]

        # compute gradient projection - step2(1)
        dot_product = (gf_bar * gu_bar).sum()
        if dot_product < 0:
            gu_bar = gu_bar - (dot_product / (gf_bar * gf_bar).sum()) * gf_bar
        grads_list = [gf_bar, gu_bar]

        proj_grads: List[paddle.Tensor] = []
        for j in range(len(self.losses)):
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

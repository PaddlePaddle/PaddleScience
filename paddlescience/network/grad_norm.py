# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn
from paddle.nn.initializer import Assign

from .network_base import NetworkBase


class GradNorm(NetworkBase):
    r"""
    Gradient normalization for adaptive loss balancing.
    Parameters:
        net(NetworkBase): The network which must have "get_shared_layer" method.
        n_loss(int): The number of loss, must be greater than 1.
        alpha(float): The hyperparameter which controls learning rate, must be greater than 0.
        weight_attr(list, tuple): The inital weights for "loss_weights". If not specified, "loss_weights" will be initialized with 1.
    """

    def __init__(self, net, n_loss, alpha, weight_attr=None):
        super().__init__()
        if not isinstance(net, NetworkBase):
            raise TypeError("'net' must be a NetworkBase subclass instance.")
        if not hasattr(net, "get_shared_layer"):
            raise TypeError("'net' must have 'get_shared_layer' method.")
        if n_loss <= 1:
            raise ValueError(
                "'n_loss' must be greater than 1, but got {}".format(n_loss)
            )
        if alpha < 0:
            raise ValueError("'alpha' is a positive number, but got {}".format(alpha))
        if weight_attr is not None:
            if len(weight_attr) != n_loss:
                raise ValueError("weight_attr must have same length with loss weights.")

        self.n_loss = n_loss
        self.net = net
        self.loss_weights = self.create_parameter(
            shape=[n_loss],
            attr=Assign(weight_attr if weight_attr else [1] * n_loss),
            dtype=self._dtype,
            is_bias=False,
        )
        self.set_grad()
        self.alpha = float(alpha)
        self.initial_losses = None

    def nn_func(self, ins):
        return self.net.nn_func(ins)

    def __getattr__(self, __name):
        try:
            return super().__getattr__(__name)
        except:
            return getattr(self.net, __name)

    def get_grad_norm_loss(self, losses):
        if isinstance(losses, list):
            losses = paddle.stack(losses)

        if self.initial_losses is None:
            self.initial_losses = losses.numpy()

        W = self.net.get_shared_layer()

        # set grad to zero
        if self.loss_weights.grad is not None:
            self.loss_weights.grad.set_value(paddle.zeros_like(self.loss_weights))

        # calulate each loss's grad
        norms = []
        for i in range(losses.shape[0]):
            grad = paddle.autograd.grad(losses[i], W, retain_graph=True)
            norms.append(paddle.norm(self.loss_weights[i] * grad[0], p=2).reshape([]))
        norms = paddle.stack(norms)

        # calculate the inverse train rate
        loss_ratio = losses.numpy() / self.initial_losses
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # calculate the mean value of grad
        mean_norm = np.mean(norms.numpy())

        # convert it to constant, instead of having grads
        constant_term = paddle.to_tensor(
            mean_norm * np.power(inverse_train_rate, self.alpha), dtype=self._dtype
        )
        # calculate the grad norm loss
        grad_norm_loss = paddle.norm(norms - constant_term, p=1)
        # update the grad of loss weights
        self.loss_weights.grad.set_value(
            paddle.autograd.grad(grad_norm_loss, self.loss_weights)[0]
        )
        #  renormalize the loss weights each step when training
        if self.training:
            self.renormalize()
        return grad_norm_loss

    def renormalize(self):
        normalize_coeff = self.n_loss / paddle.sum(self.loss_weights)
        self.loss_weights = self.create_parameter(
            shape=[self.n_loss],
            attr=Assign(self.loss_weights * normalize_coeff),
            dtype=self._dtype,
            is_bias=False,
        )
        self.set_grad()

    def reset_initial_losses(self):
        self.initial_losses = None

    def set_grad(self):
        x = paddle.ones_like(self.loss_weights)
        x *= self.loss_weights
        x.backward()
        self.loss_weights.grad.set_value(paddle.zeros_like(self.loss_weights))

    def get_weights(self):
        return self.loss_weights.numpy()

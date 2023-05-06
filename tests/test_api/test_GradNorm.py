"""
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
"""
from functools import partial

import numpy as np
import paddle
import pytest
from apibase import APIBase

import paddlescience as psci

GLOBAL_SEED = 22
np.random.seed(GLOBAL_SEED)
paddle.seed(GLOBAL_SEED)
paddle.disable_static()

loss_func = [
    paddle.sum,
    paddle.mean,
    partial(paddle.norm, p=2),
    partial(paddle.norm, p=3),
]


def randtool(dtype, low, high, shape, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


def cal_gradnorm(
    ins,
    num_ins,
    num_outs,
    num_layers,
    hidden_size,
    n_loss=3,
    alpha=0.5,
    activation="tanh",
    weight_attr=None,
):

    net = psci.network.FCNet(
        num_ins=num_ins,
        num_outs=num_outs,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
    )

    for i in range(num_layers):
        net._weights[i] = paddle.ones_like(net._weights[i])
        net._weights[i].stop_gradient = False

    grad_norm = psci.network.GradNorm(
        net=net, n_loss=n_loss, alpha=alpha, weight_attr=weight_attr
    )
    res = grad_norm.nn_func(ins)

    losses = []
    for idx in range(n_loss):
        losses.append(loss_func[idx](res).reshape([]))
    weighted_loss = grad_norm.loss_weights * paddle.stack(losses)
    loss = paddle.sum(weighted_loss)
    loss.backward(retain_graph=True)
    grad_norm_loss = grad_norm.get_grad_norm_loss(losses)
    return grad_norm_loss


class TestGradNorm(APIBase):
    def hook(self):
        """
        implement
        """
        self.types = [np.float32]
        # self.debug = True
        # enable check grad
        self.static = False
        self.enable_backward = False
        self.rtol = 1e-7


obj = TestGradNorm(cal_gradnorm)


@pytest.mark.api_network_GradNorm
def test_GradNorm0():
    xy_data = np.array([[0.1, 0.5, 0.3, 0.4, 0.2]])
    u = np.array([1.138526], dtype=np.float32)
    obj.run(res=u, ins=xy_data, num_ins=5, num_outs=3, num_layers=2, hidden_size=1)


@pytest.mark.api_network_GradNorm
def test_GradNorm1():
    xy_data = randtool("float", 0, 10, (9, 2), GLOBAL_SEED)
    u = np.array([20.636574])
    obj.run(
        res=u, ins=xy_data, num_ins=2, num_outs=3, num_layers=2, hidden_size=1, n_loss=4
    )


@pytest.mark.api_network_GradNorm
def test_GradNorm2():
    xy_data = randtool("float", 0, 1, (9, 3), GLOBAL_SEED)
    u = np.array([7.633053])
    obj.run(
        res=u,
        ins=xy_data,
        num_ins=3,
        num_outs=1,
        num_layers=2,
        hidden_size=1,
        activation="sigmoid",
    )


@pytest.mark.api_network_GradNorm
def test_GradNorm3():
    xy_data = randtool("float", 0, 1, (9, 4), GLOBAL_SEED)
    u = np.array([41.803569])
    obj.run(
        res=u,
        ins=xy_data,
        num_ins=4,
        num_outs=3,
        num_layers=2,
        hidden_size=10,
        activation="sigmoid",
        n_loss=2,
        alpha=0.2,
    )


@pytest.mark.api_network_GradNorm
def test_GradNorm4():
    xy_data = randtool("float", 0, 1, (9, 5), GLOBAL_SEED)
    u = np.array([12.606881])
    obj.run(
        res=u,
        ins=xy_data,
        num_ins=5,
        num_outs=1,
        num_layers=3,
        hidden_size=2,
        weight_attr=[1.0, 2.0, 3.0],
    )

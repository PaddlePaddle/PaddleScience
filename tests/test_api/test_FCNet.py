"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddlescience as psci
import pytest
import paddle
from apibase import APIBase
from apibase import randtool

np.random.seed(22)
paddle.seed(22)


def cal_FCNet(ins,
              num_ins,
              num_outs,
              num_layers,
              hidden_size,
              dtype='float64',
              activation='tanh'):
    """
    calculate FCNet api
    """
    net = psci.network.FCNet(
        num_ins=num_ins,
        num_outs=num_outs,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dtype=dtype,
        activation=activation)

    for i in range(num_layers):
        net.weights[i] = paddle.ones_like(net.weights[i])
    res = net.nn_func(ins)
    return res


def cal_with_np(ins,
                num_ins,
                num_outs,
                num_layers,
                hidden_size,
                activation='tanh'):
    """
    calculate with numpy
    """
    w = []
    for i in range(num_layers):
        if i == 0:
            lsize = num_ins
            rsize = hidden_size
        elif i == (num_layers - 1):
            lsize = hidden_size
            rsize = num_outs
        else:
            lsize = hidden_size
            rsize = hidden_size
        w.append(np.ones((lsize, rsize)))

    u = ins
    for i in range(num_layers - 1):
        u = np.matmul(u, w[i])
        if activation == 'tanh':
            u = np.tanh(u)
        elif activation == 'sigmoid':
            u = 1 / (1 + np.exp(-u))
    u = np.matmul(u, w[-1])
    return u


class TestFCNet(APIBase):
    """
    test flatten
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        # self.debug = True
        # enable check grad
        self.static = False


obj = TestFCNet(cal_FCNet)


@pytest.mark.api_network_FCNet
def test_FCNet0():
    """
    default
    """
    xy_data = np.array([[0.1, 0.5]])
    u = cal_with_np(xy_data, 2, 1, 2, 1)
    obj.run(res=u,
            ins=xy_data,
            num_ins=2,
            num_outs=1,
            num_layers=2,
            hidden_size=1)


@pytest.mark.api_network_FCNet
def test_FCNet1():
    """
    xy shape (9, 2)
    """
    xy_data = randtool("float", 0, 10, (9, 2))
    u = cal_with_np(xy_data, 2, 1, 2, 1)
    obj.run(res=u,
            ins=xy_data,
            num_ins=2,
            num_outs=1,
            num_layers=2,
            hidden_size=1)


@pytest.mark.api_network_FCNet
def test_FCNet2():
    """
    xy shape (9, 3)
    """
    xy_data = randtool("float", 0, 1, (9, 3))
    u = cal_with_np(xy_data, 3, 1, 2, 1)
    obj.run(res=u,
            ins=xy_data,
            num_ins=3,
            num_outs=1,
            num_layers=2,
            hidden_size=1)


@pytest.mark.api_network_FCNet
def test_FCNet3():
    """
    xy shape (9, 4)
    """
    xy_data = randtool("float", 0, 1, (9, 4))
    u = cal_with_np(xy_data, 4, 1, 2, 1)
    obj.run(res=u,
            ins=xy_data,
            num_ins=4,
            num_outs=1,
            num_layers=2,
            hidden_size=1)


@pytest.mark.api_network_FCNet
def test_FCNet4():
    """
    xy shape (9, 4)
    num_outs: 2
    """
    xy_data = randtool("float", 0, 1, (9, 4))
    u = cal_with_np(xy_data, 4, 2, 2, 1)
    obj.run(res=u,
            ins=xy_data,
            num_ins=4,
            num_outs=2,
            num_layers=2,
            hidden_size=1)


@pytest.mark.api_network_FCNet
def test_FCNet5():
    """
    xy shape (9, 4)
    num_outs: 3
    """
    xy_data = randtool("float", 0, 1, (9, 4))
    u = cal_with_np(xy_data, 4, 3, 2, 1)
    obj.run(res=u,
            ins=xy_data,
            num_ins=4,
            num_outs=3,
            num_layers=2,
            hidden_size=1)


@pytest.mark.api_network_FCNet
def test_FCNet6():
    """
    xy shape (9, 4)
    num_outs: 3
    hidden_size: 20
    """
    xy_data = randtool("float", 0, 1, (9, 4))
    u = cal_with_np(xy_data, 4, 3, 2, 20)
    obj.run(res=u,
            ins=xy_data,
            num_ins=4,
            num_outs=3,
            num_layers=2,
            hidden_size=20)


@pytest.mark.api_network_FCNet
def test_FCNet7():
    """
    xy shape (9, 4)
    num_outs: 3
    hidden_size: 20
    num_layers: 5
    """
    xy_data = randtool("float", 0, 1, (9, 4))
    u = cal_with_np(xy_data, 4, 3, 5, 20)
    obj.run(res=u,
            ins=xy_data,
            num_ins=4,
            num_outs=3,
            num_layers=5,
            hidden_size=20)


@pytest.mark.api_network_FCNet
def test_FCNet8():
    """
    xy shape (9, 4)
    num_outs: 3
    hidden_size: 20
    num_layers: 5
    activation='sigmoid'
    """
    xy_data = randtool("float", 0, 1, (9, 4))
    u = cal_with_np(xy_data, 4, 3, 5, 20, activation='sigmoid')
    obj.run(res=u,
            ins=xy_data,
            num_ins=4,
            num_outs=3,
            num_layers=5,
            hidden_size=20)

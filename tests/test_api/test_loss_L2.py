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

import paddle
import paddlescience as psci
import numpy as np
from backward import DifferenceAppro
import pytest


class TestLossL2(DifferenceAppro):
    """
    test loss_L2
    """

    def hook(self):
        """
        hook
        """
        self.debug = False


def cal_lossL2(pdes, geo, net, **kwargs):
    """
    calculate loss_L2 api
    """
    loss = psci.loss.L2(pdes=pdes, geo=geo, **kwargs)
    loss.set_batch_size(geo.get_domain_size())
    geo.to_tensor()
    pdes.to_tensor()
    r = loss.batch_run(net, 0)
    return r[0]


obj = TestLossL2(cal_lossL2)


@pytest.mark.api_loss_L2
def test_loss_L2_0():
    """
    equation: Laplace 2D
    """
    pdes = psci.pde.Laplace2D()
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(3, 3))
    bc_value = np.zeros((8, 1), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value)
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=1,
        num_layers=2,
        hidden_size=20,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    obj.run(pdes=pdes, geo=geo, net=net)


@pytest.mark.api_loss_L2
def test_loss_L2_1():
    """
    equation: Laplace 2D
    add aux_func
    """
    pdes = psci.pde.Laplace2D()
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(3, 3))
    bc_value = np.zeros((8, 1), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value)
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=1,
        num_layers=2,
        hidden_size=20,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    def RighthandBatch(xy):
        return [
            8.0 * 3.1415926 * 3.1415926 * paddle.sin(2.0 * np.pi * xy[:, 0]) *
            paddle.cos(2.0 * np.pi * xy[:, 1])
        ]

    obj.run(pdes=pdes, geo=geo, net=net, aux_func=RighthandBatch)


@pytest.mark.api_loss_L2
def test_loss_L2_2():
    """
    equation: Laplace 2D
    add aux_func
    set eq_weight
    """
    pdes = psci.pde.Laplace2D()
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(3, 3))
    bc_value = np.zeros((8, 1), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value)
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=1,
        num_layers=2,
        hidden_size=20,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    def RighthandBatch(xy):
        return [
            8.0 * 3.1415926 * 3.1415926 * paddle.sin(2.0 * np.pi * xy[:, 0]) *
            paddle.cos(2.0 * np.pi * xy[:, 1])
        ]

    obj.run(pdes=pdes,
            geo=geo,
            net=net,
            aux_func=RighthandBatch,
            eq_weight=0.5)


@pytest.mark.api_loss_L2
def test_loss_L2_3():
    """
    equation: Laplace 2D
    add aux_func
    set eq_weight
    set bc_weight
    """
    pdes = psci.pde.Laplace2D()
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(3, 3))
    bc_value = np.zeros((8, 1), dtype='float32')
    bc_weight = np.random.rand(8, 1)
    pdes.set_bc_value(bc_value=bc_value)
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=1,
        num_layers=2,
        hidden_size=20,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    def RighthandBatch(xy):
        return [
            8.0 * 3.1415926 * 3.1415926 * paddle.sin(2.0 * np.pi * xy[:, 0]) *
            paddle.cos(2.0 * np.pi * xy[:, 1])
        ]

    obj.run(pdes=pdes,
            geo=geo,
            net=net,
            aux_func=RighthandBatch,
            eq_weight=0.5,
            bc_weight=bc_weight)


@pytest.mark.api_loss_L2
def test_loss_L2_4():
    """
    equation: Laplace 2D
    add aux_func
    set eq_weight
    set bc_weight
    synthesis_method: norm
    """
    pdes = psci.pde.Laplace2D()
    geo = psci.geometry.Rectangular(
        space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(3, 3))
    bc_value = np.zeros((8, 1), dtype='float32')
    bc_weight = np.random.rand(8, 1)
    pdes.set_bc_value(bc_value=bc_value)
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=1,
        num_layers=2,
        hidden_size=20,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    def RighthandBatch(xy):
        return [
            8.0 * 3.1415926 * 3.1415926 * paddle.sin(2.0 * np.pi * xy[:, 0]) *
            paddle.cos(2.0 * np.pi * xy[:, 1])
        ]

    obj.run(pdes=pdes,
            geo=geo,
            net=net,
            aux_func=RighthandBatch,
            eq_weight=0.5,
            bc_weight=bc_weight,
            synthesis_method="norm")


@pytest.mark.api_loss_L2
def test_loss_L2_5():
    """
    equation:  NavierStokes 2D
    """
    geo = psci.geometry.Rectangular(
        space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))
    pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0)
    pdes, geo = psci.discretize(pdes, geo, space_nsteps=(4, 4))

    bc_value = np.zeros((12, 2), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])
    net = psci.network.FCNet(
        num_ins=2,
        num_outs=3,
        num_layers=2,
        hidden_size=4,
        dtype="float32",
        activation='tanh')
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    obj.run(pdes=pdes,
            geo=geo,
            net=net,
            eq_weight=0.01,
            bc_weight=None,
            synthesis_method='norm')

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

import paddle
import paddlescience as psci
import numpy as np
from backward import DifferenceAppro
import pytest


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


class TestLossL2(DifferenceAppro):
    """
    test loss_L2
    """

    def hook(self):
        """
        hook
        """
        self.debug = False
        self.time_dependent = True


obj = TestLossL2(cal_lossL2)


@pytest.mark.api_loss_L2
def test_loss_L2_0():
    """
    equation: NavierStokes 2D
    """
    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(-0.05, -0.05),
        space_extent=(0.05, 0.05))
    pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=True)
    pdes, geo = psci.discretize(pdes, geo, time_nsteps=3, space_nsteps=(3, 3))
    bc_value = np.zeros((24, 2), dtype='float32')
    ic_value = np.zeros((9, 2), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])
    pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1])

    net = psci.network.FCNet(
        num_ins=3,
        num_outs=3,
        num_layers=2,
        hidden_size=5,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    obj.run(pdes=pdes, geo=geo, net=net)


@pytest.mark.api_loss_L2
def test_loss_L2_1():
    """
    equation: NavierStokes 2D
    eq_weight=0.01
    """
    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(-0.05, -0.05),
        space_extent=(0.05, 0.05))
    pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=True)
    pdes, geo = psci.discretize(pdes, geo, time_nsteps=3, space_nsteps=(3, 3))
    bc_value = np.zeros((24, 2), dtype='float32')
    ic_value = np.zeros((9, 2), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])
    pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1])

    net = psci.network.FCNet(
        num_ins=3,
        num_outs=3,
        num_layers=2,
        hidden_size=5,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])

    obj.run(pdes=pdes, geo=geo, net=net, eq_weight=0.01)


@pytest.mark.api_loss_L2
def test_loss_L2_2():
    """
    equation: NavierStokes 2D
    eq_weight=0.01
    set bc_weight
    """
    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(-0.05, -0.05),
        space_extent=(0.05, 0.05))
    pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=True)
    pdes, geo = psci.discretize(pdes, geo, time_nsteps=3, space_nsteps=(3, 3))
    bc_value = np.zeros((24, 2), dtype='float32')
    ic_value = np.zeros((9, 2), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])
    pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1])

    net = psci.network.FCNet(
        num_ins=3,
        num_outs=3,
        num_layers=2,
        hidden_size=5,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])
    bc_weight = np.ones((24, 2))
    obj.run(pdes=pdes, geo=geo, net=net, eq_weight=0.01, bc_weight=bc_weight)


@pytest.mark.api_loss_L2
def test_loss_L2_3():
    """
    equation: NavierStokes 2D
    eq_weight=0.01
    set bc_weight
    synthesis_method=norm
    """
    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(-0.05, -0.05),
        space_extent=(0.05, 0.05))
    pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=True)
    pdes, geo = psci.discretize(pdes, geo, time_nsteps=3, space_nsteps=(3, 3))
    bc_value = np.zeros((24, 2), dtype='float32')
    ic_value = np.zeros((9, 2), dtype='float32')
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])
    pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1])

    net = psci.network.FCNet(
        num_ins=3,
        num_outs=3,
        num_layers=2,
        hidden_size=5,
        dtype="float32",
        activation="tanh")
    for i in range(2):
        net.weights[i] = paddle.ones_like(net.weights[i])
    bc_weight = np.ones((24, 2))
    obj.run(pdes=pdes,
            geo=geo,
            net=net,
            eq_weight=0.01,
            bc_weight=bc_weight,
            synthesis_method='norm')

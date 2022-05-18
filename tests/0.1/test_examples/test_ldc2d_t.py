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

import pytest
import paddle
import paddlescience as psci
import numpy as np


def GenBC(txy, bc_index):
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(txy[id][2] - 0.05) < 1e-4:
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value


# Generate BC weight
def GenBCWeight(txy, bc_index):
    bc_weight = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(txy[id][2] - 0.05) < 1e-4:
            bc_weight[i][0] = 1.0 - 20 * abs(txy[id][1])
            bc_weight[i][1] = 1.0
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
    return bc_weight


# Generate IC value
def GenIC(txy, ic_index):
    ic_value = np.zeros((len(ic_index), 2)).astype(np.float32)
    for i in range(len(ic_index)):
        id = ic_index[i]
        if abs(txy[id][2] - 0.05) < 1e-4:
            ic_value[i][0] = 1.0
            ic_value[i][1] = 0.0
        else:
            ic_value[i][0] = 0.0
            ic_value[i][1] = 0.0
    return ic_value


# Geometry
geo = psci.geometry.Rectangular(
    time_dependent=True,
    time_origin=0,
    time_extent=0.5,
    space_origin=(-0.05, -0.05),
    space_extent=(0.05, 0.05))

# PDE Laplace
pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=True)

# Discretization
pdes, geo = psci.discretize(pdes, geo, time_nsteps=6, space_nsteps=(3, 3))

# bc value
bc_value = GenBC(geo.get_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# ic value
ic_value = GenIC(geo.get_domain(), geo.get_ic_index())
pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=3,
    num_outs=3,
    num_layers=10,
    hidden_size=50,
    dtype="float32",
    activation='tanh')

for i in range(10):
    net.weights[i] = paddle.ones_like(net.weights[i])

# Loss, TO rename
bc_weight = GenBCWeight(geo.domain, geo.bc_index)
loss = psci.loss.L2(pdes=pdes,
                    geo=geo,
                    eq_weight=0.01,
                    bc_weight=bc_weight,
                    synthesis_method='norm')

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=1)

# Use solution
rslt_t = solution(geo)
# print(rslt_t)
# Get the result of last moment
rslt = rslt_t[(-geo.space_domain_size):, :]


@pytest.mark.ldc2d_t
def test_ldc2d_t():
    """
    test ldc2d_t
    """
    golden = np.load("./golden/ldc2d_t.npz")
    assert np.allclose(rslt_t, golden["expect_t"])
    assert np.allclose(rslt, golden["expect"])

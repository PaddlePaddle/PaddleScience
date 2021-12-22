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

import pytest
import paddlescience as psci
import numpy as np
import paddle


# Generate BC value
def GenBC(xy, bc_index):
    """
    GenBC
    """
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    length1 = len(bc_index)
    for i in range(length1):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-4:
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value


# Generate BC weight
def GenBCWeight(xy, bc_index):
    """
    GenBCWeight
    """
    bc_weight = np.zeros((len(bc_index), 2)).astype(np.float32)
    length2 = len(bc_index)
    for i in range(length2):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-4:
            bc_weight[i][0] = 1.0 - 20 * abs(xy[id][0])
            bc_weight[i][1] = 1.0
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
    return bc_weight


# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))

# PDE Laplace
pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0)

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(4, 4))

# bc value
bc_value = GenBC(geo.space_domain, geo.bc_index)
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=2,
    hidden_size=2,
    dtype="float32",
    activation="tanh")

net._parameters["w_0"].set_value(
    paddle.to_tensor([[1, 1], [1, 1]]).astype("float32"))
net._parameters["w_1"].set_value(
    paddle.to_tensor([[1, 1, 1], [1, 1, 1]]).astype("float32"))
print(net._parameters)

# Loss, TO rename
bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
loss = psci.loss.L2(pdes=pdes,
                    geo=geo,
                    eq_weight=0.01,
                    bc_weight=bc_weight,
                    synthesis_method="norm")

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=15)

# Use solution
rslt = solution(geo)
#print(paddle.to_tensor(rslt))

# Infer with another geometry
geo_1 = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))
geo_1 = geo_1.discretize(space_nsteps=(3, 3))
rslt_1 = solution(geo_1)
#print(paddle.to_tensor(rslt_1))


@pytest.mark.ldc2d
def test_Ldc2D():
    """
    test Ldc2D
    """
    expect1 = np.array([
        [-0.15440597, -0.17988314, -0.16472158],
        [-0.08826673, -0.11550318, -0.10034302],
        [-0.02190595, -0.05090757, -0.03574881],
        [0.04453391, 0.01376501, 0.02892237],
        [-0.08822407, -0.11546165, -0.10030149],
        [-0.02186322, -0.05086598, -0.03570721],
        [0.04457666, 0.01380662, 0.02896398],
        [0.11095245, 0.07841684, 0.09357280],
        [-0.02182047, -0.05082437, -0.03566560],
        [0.04461941, 0.01384824, 0.02900560],
        [0.11099510, 0.07845836, 0.09361432],
        [0.17716406, 0.14286724, 0.15802181],
        [0.04466216, 0.01388985, 0.02904721],
        [0.11103776, 0.07849988, 0.09365584],
        [0.17720655, 0.14290859, 0.15806317],
        [0.24302775, 0.20697896, 0.22213215],
    ])
    expect2 = np.array([
        [-0.15440597, -0.17988314, -0.16472158],
        [-0.05510515, -0.08322369, -0.06806422],
        [0.04453391, 0.01376501, 0.02892237],
        [-0.05504109, -0.08316133, -0.06800187],
        [0.04459803, 0.01382742, 0.02898479],
        [0.14409299, 0.11067586, 0.12583113],
        [0.04466216, 0.01388985, 0.02904721],
        [0.14415684, 0.11073800, 0.12589327],
        [0.24302775, 0.20697896, 0.22213215],
    ])
    assert np.allclose(rslt, expect1), "the rslt was changed"
    assert np.allclose(rslt_1, expect2), "the rslt_1 was changed"

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

import paddlescience as psci
import numpy as np
import paddle

paddle.seed(1)
np.random.seed(1)

# paddle.enable_static()
# paddle.disable_static()

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.05, -0.05), extent=(0.05, 0.05))

geo.add_boundary(name="top", criteria=lambda x, y: abs(y - 0.05) < 1e-5)
geo.add_boundary(name="down", criteria=lambda x, y: abs(y + 0.05) < 1e-5)
geo.add_boundary(name="left", criteria=lambda x, y: abs(x + 0.05) < 1e-5)
geo.add_boundary(name="right", criteria=lambda x, y: abs(x - 0.05) < 1e-5)

# discretize geometry
npoints = 10201
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# N-S
pde = psci.pde.NavierStokes(
    nu=0.01, rho=1.0, dim=2, time_dependent=True, weight=0.0001)
pde.set_time_interval([0.0, 0.5])

# set bounday condition
weight_top_u = lambda x, y: 1.0 - 20.0 * abs(x)
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0, weight=weight_top_u)
bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)

# add bounday and boundary condition
pde.add_bc("top", bc_top_u, bc_top_v)
pde.add_bc("down", bc_down_u, bc_down_v)
pde.add_bc("left", bc_left_u, bc_left_v)
pde.add_bc("right", bc_right_u, bc_right_v)

# add initial condition
ic_u = psci.ic.IC('u', rhs=0.0)
ic_v = psci.ic.IC('v', rhs=0.0)
pde.add_ic(ic_u, ic_v)

# discretization pde
pde_disc = pde.discretize(time_step=0.1, geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=3, num_outs=3, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=10000)

psci.visu.save_vtk(
    time_array=pde_disc.time_array, geo_disc=pde_disc.geometry, data=solution)

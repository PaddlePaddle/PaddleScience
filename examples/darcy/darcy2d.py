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

paddle.enable_static()
# paddle.disable_static()

# analytical solution 
analytic_sol = lambda x, y: np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

# analytical rhs
analytic_rhs = lambda x, y: 8.0 * np.pi**2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))

geo.add_boundary(name="top", criteria=lambda x, y: y == 1.0)
geo.add_boundary(name="down", criteria=lambda x, y: y == 0.0)
geo.add_boundary(name="left", criteria=lambda x, y: x == 0.0)
geo.add_boundary(name="right", criteria=lambda x, y: x == 1.0)

# Poisson
pde = psci.pde.Poisson(dim=2, rhs=analytic_rhs)

# set bounday condition
bc_top = psci.bc.Dirichlet('u', rhs=analytic_sol)
bc_down = psci.bc.Dirichlet('u', rhs=analytic_sol)
bc_left = psci.bc.Dirichlet('u', rhs=analytic_sol)
bc_right = psci.bc.Dirichlet('u', rhs=analytic_sol)

pde.add_geometry(geo)

# add bounday and boundary condition
pde.add_bc("top", bc_top)
pde.add_bc("down", bc_down)
pde.add_bc("left", bc_left)
pde.add_bc("right", bc_right)

# discretization
pde = psci.discretize(pde, space_npoints=10000, space_method="uniform")

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=10,
    hidden_size=50,
    dtype="float32",
    activation='tanh')

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)
solution = solver.solve(num_epoch=10000)

psci.visu.save_vtk(geo_disc=pde.geometry, data=solution)

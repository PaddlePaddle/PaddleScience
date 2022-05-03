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
paddle.disable_static()

# constants
Re = 40.0
r = Re / 2 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)

# Kovasznay solution
ref_sol_u = lambda x, y: 1 - np.exp(r * x) * np.cos(2 * np.pi * y)
ref_sol_v = lambda x, y: r / (2 * np.pi) * np.exp(r * x) * np.sin(2 * np.pi * y)
ref_sol_p = lambda x, y: 1 / 2 - 1 / 2 * np.exp(2 * r * x)

geo = psci.geometry.Rectangular(origin=(-0.5, -0.5), extent=(1.5, 1.5))

# geo.add_boundary(name="boarder", criteria=lambda x, y: x == -0.5 or x==1.5 or y==-0.5 or y ==1.5)

geo.add_boundary(name="top", criteria=lambda x, y: y == -0.5)
geo.add_boundary(name="down", criteria=lambda x, y: y == 1.5)
geo.add_boundary(name="left", criteria=lambda x, y: x == -0.5)
geo.add_boundary(name="right", criteria=lambda x, y: x == 1.5)

# N-S equation
pde = psci.pde.NavierStokes(nu=1.0 / Re, rho=1.0, dim=2)

# set boundary condition
bc_border_u = psci.bc.Dirichlet('u', ref_sol_u)
bc_border_v = psci.bc.Dirichlet('v', ref_sol_v)
bc_border_p = psci.bc.Dirichlet('p', ref_sol_p)

pde.add_geometry(geo)

# add bounday and boundary condition
pde.add_bc("top", bc_border_u)
pde.add_bc("top", bc_border_v)
pde.add_bc("top", bc_border_p)
pde.add_bc("down", bc_border_u)
pde.add_bc("down", bc_border_v)
pde.add_bc("down", bc_border_p)
pde.add_bc("left", bc_border_u)
pde.add_bc("left", bc_border_v)
pde.add_bc("left", bc_border_p)
pde.add_bc("right", bc_border_u)
pde.add_bc("right", bc_border_v)
pde.add_bc("right", bc_border_p)

# discretization
npoints = 2601
pde_disc = psci.discretize(pde, space_npoints=npoints, space_method="uniform")

net = psci.network.FCNet(
    num_ins=2, num_outs=3, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=10)

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)

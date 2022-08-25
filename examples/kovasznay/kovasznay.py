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

cfg = psci.utils.parse_args()

if cfg is not None:
    # Geometry
    npoints = cfg['Geometry']['npoints']
    seed_num = cfg['Geometry']['seed']
    sampler_method = cfg['Geometry']['sampler_method']
    # Network
    epochs = cfg['Global']['epochs']
    num_layers = cfg['Model']['num_layers']
    hidden_size = cfg['Model']['hidden_size']
    activation = cfg['Model']['activation']
    # Optimizer
    learning_rate = cfg['Optimizer']['lr']['learning_rate']
    # Post-processing
    solution_filename = cfg['Post-processing']['solution_filename']
    vtk_filename = cfg['Post-processing']['vtk_filename']
    checkpoint_path = cfg['Post-processing']['checkpoint_path']
else:
    # Geometry
    npoints = 2601
    seed_num = 1
    sampler_method = 'uniform'
    # Network
    epochs = 10000
    num_layers = 10
    hidden_size = 50
    activation = 'tanh'
    # Optimizer
    learning_rate = 0.001
    # Post-processing
    solution_filename = 'output_kovasznayd'
    vtk_filename = 'output_kovasznay'
    checkpoint_path = 'checkpoints'

paddle.seed(seed_num)
np.random.seed(seed_num)

# constants
Re = 40.0
r = Re / 2 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)

# Kovasznay solution
ref_sol_u = lambda x, y: 1.0 - np.exp(r * x) * np.cos(2.0 * np.pi * y)
ref_sol_v = lambda x, y: r / (2 * np.pi) * np.exp(r * x) * np.sin(2.0 * np.pi * y)
ref_sol_p = lambda x, y: 1.0 / 2.0 - 1.0 / 2.0 * np.exp(2.0 * r * x)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.5, -0.5), extent=(1.5, 1.5))
geo.add_boundary(
    name="boarder",
    criteria=lambda x, y: (x == -0.5) | (x == 1.5) | (y == -0.5) | (y == 1.5))

# discretization
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)

# N-S equation
pde = psci.pde.NavierStokes(nu=1.0 / Re, rho=1.0, dim=2)

# set boundary condition
bc_border_u = psci.bc.Dirichlet('u', ref_sol_u)
bc_border_v = psci.bc.Dirichlet('v', ref_sol_v)
bc_border_p = psci.bc.Dirichlet('p', ref_sol_p)

# add bounday and boundary condition
pde.add_bc("boarder", bc_border_u)
pde.add_bc("boarder", bc_border_v)
pde.add_bc("boarder", bc_border_p)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(
    learning_rate=learning_rate, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=epochs)

psci.visu.save_vtk(
    filename=vtk_filename, geo_disc=pde_disc.geometry, data=solution)

psci.visu.save_npy(
    filename=solution_filename, geo_disc=pde_disc.geometry, data=solution)

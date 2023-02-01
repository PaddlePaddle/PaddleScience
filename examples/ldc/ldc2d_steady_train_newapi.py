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
    npoints = 10201
    seed_num = 1
    sampler_method = 'uniform'
    # Network
    epochs = 20000
    num_layers = 10
    hidden_size = 50
    activation = 'tanh'
    # Optimizer
    learning_rate = 0.001
    # Post-processing
    solution_filename = 'output_ldc2d_steady_train'
    vtk_filename = 'output_ldc2d_steady_train'
    checkpoint_path = 'checkpoints'

paddle.seed(seed_num)
np.random.seed(seed_num)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.05, -0.05), extent=(0.05, 0.05))

def boundary_top(x, y):
    return abs(y - 0.05) < 1e-5
def boundary_down(x, y):
    return abs(y + 0.05) < 1e-5
def boundary_left(x, y):
    return abs(x + 0.05) < 1e-5
def boundary_right(x, y):
    return abs(x - 0.05) < 1e-5
geo.add_boundary(name="top", criteria=boundary_top)
geo.add_boundary(name="down", criteria=boundary_down)
geo.add_boundary(name="left", criteria=boundary_left)
geo.add_boundary(name="right", criteria=boundary_right)
# geo.add_boundary(name="top", criteria=lambda x,y: abs(y - 0.05) < 1e-5)
# geo.add_boundary(name="down", criteria=lambda x,y: abs(y + 0.05) < 1e-5)
# geo.add_boundary(name="left", criteria=lambda x,y: abs(x + 0.05) < 1e-5)
# geo.add_boundary(name="right", criteria=lambda x,y: abs(x - 0.05) < 1e-5)

# discretize geometry
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)

# N-S: Re = 100, u = 1.0, nu = rho u d / Re = 1.0 * 1.0 * 0.1 / 100 = 0.001
pde = psci.pde.NavierStokes(
    nu=0.001, rho=1.0, dim=2, time_dependent=False, weight=0.0001)

# define bounday conditions
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0)
# weight_top_u = lambda x, y: 1.0 - 20.0 * abs(x)
# bc_top_u = psci.bc.Dirichlet('u', rhs=1.0, weight=weight_top_u)
bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)

# set boundary conditions
pde.set_bc("top", bc_top_u, bc_top_v)
pde.set_bc("down", bc_down_u, bc_down_v)
pde.set_bc("left", bc_left_u, bc_left_v)
pde.set_bc("right", bc_right_u, bc_right_v)
# pde.add_bc("top", bc_top_u, bc_top_v)
# pde.add_bc("down", bc_down_u, bc_down_v)
# pde.add_bc("left", bc_left_u, bc_left_v)
# pde.add_bc("right", bc_right_u, bc_right_v)

# discretization pde
# pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

# eq loss
inputeq = geo_disc.interior
outeq = net(inputeq)
losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
# bc loss
inputbc_top = geo_disc.boundary["top"];     outbc_top = net(inputbc_top)
lossbc_top = psci.loss.BcLoss("top", netout=outbc_top)
inputbc_down = geo_disc.boundary["down"];   outbc_down = net(inputbc_down)
lossbc_down = psci.loss.BcLoss("down", netout=outbc_down)
inputbc_left = geo_disc.boundary["left"];   outbc_left = net(inputbc_left)
lossbc_left = psci.loss.BcLoss("left", netout=outbc_left)
inputbc_right = geo_disc.boundary["right"]; outbc_right = net(inputbc_right)
lossbc_right = psci.loss.BcLoss("right", netout=outbc_right)

# total loss
loss = losseq1 + losseq2 + losseq3 + \
    10*lossbc_top + 10*lossbc_down + 10*lossbc_left + 10*lossbc_right

# Loss
# loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(
    learning_rate=learning_rate, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)
# solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=epochs)

# Save result to vtk
psci.visu.__save_vtk_raw(filename="output_s_train", cordinate=geo_disc, data=solution)
# psci.visu.save_vtk(
#     filename=vtk_filename, geo_disc=pde_disc.geometry, data=solution)

# psci.visu.save_npy(
#     filename=solution_filename, geo_disc=pde_disc.geometry, data=solution)

# # MSE
# # TODO: solution array to dict: interior, bc
# cord = pde_disc.geometry.interior
# ref = ref_sol(cord[:, 0], cord[:, 1])
# mse2 = np.linalg.norm(solution[0][:, 0] - ref, ord=2)**2

# n = 1
# for cord in pde_disc.geometry.boundary.values():
#     ref = ref_sol(cord[:, 0], cord[:, 1])
#     mse2 += np.linalg.norm(solution[n][:, 0] - ref, ord=2)**2
#     n += 1

# mse = mse2 / npoints

# print("MSE is: ", mse)

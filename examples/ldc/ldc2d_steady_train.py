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

paddle.seed(42)
np.random.seed(42)

# set geometry and boundary
# geo = psci.geometry.Rectangular(origin=(-0.05, -0.05), extent=(0.05, 0.05))
geo = psci.neo_geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))
geo.add_sample_config("interior", 9801)
geo.add_sample_config("boundary_top", 101)
geo.add_sample_config("boundary_down", 101)
geo.add_sample_config("boundary_left", 99)
geo.add_sample_config("boundary_right", 99)
points_dict = geo.fetch_batch_data()
# for k, v in points_dict.items():
#     print(f"{k} {len(v)}")
# exit()
# geo.add_boundary(name="top", criteria=lambda x, y: abs(y - 0.05) < 1e-5)
# geo.add_boundary(name="down", criteria=lambda x, y: abs(y + 0.05) < 1e-5)
# geo.add_boundary(name="left", criteria=lambda x, y: abs(x + 0.05) < 1e-5)
# geo.add_boundary(name="right", criteria=lambda x, y: abs(x - 0.05) < 1e-5)

# discretize geometry
# geo_disc = geo.discretize(npoints=npoints, method=sampler_method)
geo_disc = geo
geo_disc.interior = points_dict["interior"]
geo_disc.boundary = {
    "top": points_dict["boundary_top"],
    "down": points_dict["boundary_down"],
    "left": points_dict["boundary_left"],
    "right": points_dict["boundary_right"]
}
geo_disc.user = None
geo_disc.normal = {
    "top": None,
    "down": None,
    "left": None,
    "right": None
}
"""
interior (9801, 2)
top (101, 2)
down (101, 2)
left (99, 2)
right (99, 2)
"""
# N-S
pde = psci.pde.NavierStokes(
    nu=0.01, rho=1.0, dim=2, time_dependent=False, weight=0.0001)

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
neumann_bc_right_u = psci.bc.Neumann('u', rhs=0.0)
# print(geo_disc.boundary["right"][1].shape)
# print(geo_disc.boundary["right"][1][:, 0].mean())
# print(geo_disc.boundary["right"][1][:, 1].mean())

# add bounday and boundary condition
pde.add_bc("top", bc_top_u, bc_top_v)
pde.add_bc("down", bc_down_u, bc_down_v)
pde.add_bc("left", bc_left_u, bc_left_v)
# pde.add_bc("right", bc_right_u, bc_right_v)
pde.add_bc("right", neumann_bc_right_u)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

# Loss
loss = psci.loss.L2(p=2)

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

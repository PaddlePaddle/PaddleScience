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
    epochs = 20  #000
    num_layers = 5
    hidden_size = 20
    activation = 'tanh'
    # Optimizer
    learning_rate = 0.001
    # Post-processing
    solution_filename = 'output_laplace2d'
    vtk_filename = 'output_laplace2d'
    checkpoint_path = 'checkpoints'

paddle.enable_static()
paddle.seed(seed_num)
np.random.seed(seed_num)

# analytical solution 
ref_sol = lambda x, y: np.cos(x) * np.cosh(y)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
geo.add_boundary(
    name="around",
    criteria=lambda x, y: (y == 1.0) | (y == 0.0) | (x == 0.0) | (x == 1.0))

# discretize geometry
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)

# Laplace
pde = psci.pde.Laplace(dim=2)

# set bounday condition
bc_around = psci.bc.Dirichlet('u', rhs=ref_sol)

# add bounday and boundary condition
pde.add_bc("around", bc_around)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Lbfgs()

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=epochs)

psci.visu.save_vtk(
    filename=vtk_filename, geo_disc=pde_disc.geometry, data=solution)

psci.visu.save_npy(
    filename=solution_filename, geo_disc=pde_disc.geometry, data=solution)

# MSE
# TODO: solution array to dict: interior, bc
cord = pde_disc.geometry.interior
ref = ref_sol(cord[:, 0], cord[:, 1])
mse2 = np.linalg.norm(solution[0][:, 0] - ref, ord=2)**2

n = 1
for cord in pde_disc.geometry.boundary.values():
    ref = ref_sol(cord[:, 0], cord[:, 1])
    mse2 += np.linalg.norm(solution[n][:, 0] - ref, ord=2)**2
    n += 1

mse = mse2 / npoints

print("MSE is: ", mse)

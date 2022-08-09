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
    npoints = 60000
    seed_num = 1
    sampler_method = "sampling"
    # Network
    epochs = 2000
    num_layers = 10
    hidden_size = 50
    activation = 'tanh'
    # Optimizer
    learning_rate = 0.001
    # Post-processing
    solution_filename = 'output_cylinder3d_steady'
    vtk_filename = 'output_cylinder3d_steady'
    checkpoint_path = 'checkpoints'

paddle.seed(seed_num)
np.random.seed(seed_num)

# load real data
real_data = np.load("./re20_5.0.npy").astype("float32")
real_cord = real_data[:, 0:3]
real_sol = real_data[:, 3:7]

cc = (0.0, 0.0)
cr = 0.5
geo = psci.geometry.CylinderInCube(
    origin=(-8, -8, -0.5),
    extent=(25, 8, 0.5),
    circle_center=cc,
    circle_radius=cr)

geo.add_boundary(name="left", criteria=lambda x, y, z: abs(x + 8.0) < 1e-4)
geo.add_boundary(name="right", criteria=lambda x, y, z: abs(x - 25.0) < 1e-4)
geo.add_boundary(
    name="circle",
    criteria=lambda x, y, z: ((x - cc[0])**2 + (y - cc[1])**2 - cr**2) < 1e-4)

# discretize geometry
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)
geo_disc.user = real_cord

# N-S
pde = psci.pde.NavierStokes(
    nu=0.05,
    rho=1.0,
    dim=3,
    time_dependent=False,
    weight=[4.0, 0.01, 0.01, 0.01])

# boundary condition on left side: u=1, v=w=0
bc_left_u = psci.bc.Dirichlet('u', rhs=1.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_w = psci.bc.Dirichlet('w', rhs=0.0)

# boundary condition on right side: p=0
bc_right_p = psci.bc.Dirichlet('p', rhs=0.0)

# boundary on circle
bc_circle_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_circle_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_circle_w = psci.bc.Dirichlet('w', rhs=0.0)

# add bounday and boundary condition
pde.add_bc("left", bc_left_u, bc_left_v, bc_left_w)
pde.add_bc("right", bc_right_p)
pde.add_bc("circle", bc_circle_u, bc_circle_v, bc_circle_w)

# discritize pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# network
net = psci.network.FCNet(
    num_ins=3,
    num_outs=4,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)

# loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(
    learning_rate=learning_rate, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)

solver.feed_data_user(real_sol)  # add real solution

solution = solver.solve(num_epoch=epochs)

psci.visu.save_vtk(filename=vtk_filename, geo_disc=pde.geometry, data=solution)

psci.visu.save_npy(
    filename=solution_filename, geo_disc=pde.geometry, data=solution)

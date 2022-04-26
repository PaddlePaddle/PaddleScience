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


# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.05, -0.05), extent=(0.05, 0.05))

geo.add_boundary(name="top", criteria=lambda x, y: y == 0.05)
geo.add_boundary(name="down", criteria=lambda x, y: y == -0.05)
geo.add_boundary(name="left", criteria=lambda x, y: x == -0.05)
geo.add_boundary(name="right", criteria=lambda x, y: x == 0.05)

# N-S
pde = psci.pde.NavierStokes(
    nu=0.01, rho=1.0, dim=2, time_dependent=True, weight=0.01)

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

pde.add_geometry(geo)

# add bounday and boundary condition
pde.add_bc("top", bc_top_u, bc_top_v)
pde.add_bc("down", bc_down_u, bc_down_v)
pde.add_bc("left", bc_left_u, bc_left_v)
pde.add_bc("right", bc_right_u, bc_right_v)

# discretization
pde_disc = psci.discretize(
    pde, time_step=0.1, space_npoints=25, space_method="uniform")

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=3, num_outs=3, num_layers=10, hidden_size=50, activation='tanh')

# Loss
loss = psci.loss.L2()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
solution = solver.solve(num_epoch=1)

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)

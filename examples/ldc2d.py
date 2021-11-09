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

import paddlescience as psci
import numpy as np
import paddle

# Random Seed
paddle.seed(1234)


# Generate BC value
def GenBC(xy, bc_index):
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-4:
            bc_value[i][0] = 1.0 - 20.0 * abs(xy[id][0])
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value


# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))

# PDE Laplace
pdes = psci.pde.NavierStokes2D(nu=0.01, rho=1.0)

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_steps=(11, 11))

# bc value
bc_value = GenBC(geo.steps, geo.bc_index)
# print("bc_v,shape: ", bc_value.shape)
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=5,
    hidden_size=20,
    dtype="float32",
    activation='tanh')

# Loss, TO rename
loss = psci.loss.L2(pdes=pdes, geo=geo)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=3000, batch_size=None)

# Use solution
rslt = solution(geo).numpy()
psci.visu.Rectangular2D(geo, rslt[:, 0], filename="rslt_u")
psci.visu.Rectangular2D(geo, rslt[:, 1], filename="rslt_v")
# print("rslt: ", rslt)

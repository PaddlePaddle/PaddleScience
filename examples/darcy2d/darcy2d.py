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


# Analytical solution
def DarcyRecSolution(x, y):
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


# Generate analytical Solution using Geometry points
def GenSolution(xy, bc_index):
    sol = np.zeros((len(xy), 1)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    for i in range(len(xy)):
        sol[i][0] = DarcyRecSolution(xy[i][0], xy[i][1])

    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]]

    return [sol, bc_value]


# right-hand side
def Righthand(xy):
    return 8.0 * 3.1415926 * 3.1415926 * paddle.sin(
        2.0 * np.pi * xy[0]) * paddle.cos(2.0 * np.pi * xy[1])


# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(0.0, 0.0), space_extent=(1.0, 1.0))

# PDE Laplace
pdes = psci.pde.Laplace2D()

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_steps=(41, 41))

# bc value
golden, bc_value = GenSolution(geo.space_domain, geo.bc_index)
pdes.set_bc_value(bc_value=bc_value)
psci.visu.save_vtk(geo, golden, 'golden_darcy_2d')
psci.data.save_data(golden, 'golden_darcy_2d.npy')

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=5,
    hidden_size=20,
    dtype="float32",
    activation="tanh")

# Loss, TO rename
loss = psci.loss.L2(pdes=pdes, geo=geo, aux_func=Righthand)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000, batch_size=None)

# Use solution
rslt = solution(geo).numpy()
psci.visu.save_vtk(geo, rslt, 'rslt_darcy_2d')
psci.data.save_data(rslt, 'rslt_darcy_2d.npy')

# Calculate diff and l2 relative error
diff = rslt - golden
psci.visu.save_vtk(geo, diff, 'diff_darcy_2d')
psci.data.save_data(diff, 'diff_darcy_2d.npy')
l2_relative_error = np.linalg.norm(diff, ord=2) / geo.get_nsteps()
print("l2_relative_error: ", l2_relative_error)

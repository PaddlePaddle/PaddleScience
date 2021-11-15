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
def LaplaceRecSolution(x, y, k=1.0):
    if (k == 0.0):
        return x * y
    else:
        return np.cos(k * x) * np.cosh(k * y)

# Generate analytical solution using geometry points
def GenSolution(xy, bc_index):
    sol = np.zeros(len(xy)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 1)).astype(np.float32)
    for i in range(len(xy)):
        sol[i] = LaplaceRecSolution(xy[i][0], xy[i][1])
    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]]
    return [sol, bc_value]

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

psci.visu.Rectangular2D(geo, golden, 'golden_laplace_2d')
psci.data.save_data(golden, "golden_laplace_2d.npy")

# Network
net = psci.network.FCNet(
    num_ins=2,
    num_outs=1,
    num_layers=5,
    hidden_size=20,
    dtype="float32",
    activation="tanh")

# Loss, TO rename
loss = psci.loss.L2(pdes=pdes, geo=geo)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000, batch_size=None)

# Use solution
rslt = solution(geo).numpy()
psci.visu.Rectangular2D(geo, rslt, 'rslt_laplace_2d')
psci.data.save_data(rslt, './rslt_laplace_2d.npy')

# Calculate diff and l2 relative error
diff = rslt - golden
psci.visu.Rectangular2D(geo, diff, 'diff_laplace_2d')
psci.data.save_data(diff, './diff_laplace_2d.npy')

l2_relative_error = np.linalg.norm(diff, ord=2) / geo.get_nsteps()
print("l2_relative_error: ", l2_relative_error)

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

# net = psci.network.FCNet(
#     num_ins=2, num_outs=1, num_layers=5, hidden_size=20, activation='tanh')

# net.make_network()

# w1 = paddle.nn.initializer.Constant(2.0)
# b1 = paddle.nn.initializer.Constant(3.0)
# net.initialize(n=[1,2], weight_init=w1, bias_init=b1) 

# exit()

import sympy

# analytical solution 
ref_sol = lambda x, y: np.cos(x) * np.cosh(y)

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
geo.add_boundary(
    name="around",
    criteria=lambda x, y: (y == 1.0) | (y == 0.0) | (x == 0.0) | (x == 1.0))

# discretize geometry
npoints = 16
geo_disc = geo.discretize(npoints=npoints, method="uniform")

# Laplace
pde = psci.pde.Laplace(dim=2)

# set bounday condition
bc_around = psci.bc.Dirichlet('u', rhs=ref_sol)

# add bounday and boundary condition
pde.set_bc("around", bc_around)

# discretization pde
pde_disc = pde.discretize(geo_disc=geo_disc)

# network
net = psci.network.FCNet(
    num_ins=2, num_outs=1, num_layers=5, hidden_size=20, activation='tanh')

input = np.ones((3, 2))

out = net(input)
loss = psci.loss.EqLoss(eq=pde._equations[0], output=out)

exit()

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())
# opt = psci.optimizer.Lbfgs()

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

exit()

solution = solver.solve(num_epoch=1, checkpoint_freq=20)

psci.visu.save_vtk(geo_disc=pde_disc.geometry, data=solution)

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

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

Re = 40
r = Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)


def KovasznayRecSolution(x, y):
    u = 1 - np.exp(r * x) * np.cos(2 * np.pi * y)
    v = r / (2 * np.pi) * np.exp(r * x) * np.sin(2 * np.pi * y)
    p = 1 / 2 - 1 / 2 * np.exp(2 * r * x)

    return u, v, p


# Generate BC value
def GenBC(xy, bc_index):
    sol = np.zeros((len(xy), 3)).astype(np.float32)
    bc_value = np.zeros((len(bc_index), 3)).astype(np.float32)
    for i in range(len(xy)):
        sol[i][0], sol[i][1], sol[i][2] = KovasznayRecSolution(xy[i][0],
                                                               xy[i][1])

    for i in range(len(bc_index)):
        bc_value[i][0] = sol[bc_index[i]][0]
        bc_value[i][1] = sol[bc_index[i]][1]
        bc_value[i][2] = sol[bc_index[i]][2]

    return bc_value


geo = psci.geometry.Rectangular(
    space_origin=(-0.5, -0.5), space_extent=(1.5, 1.5))

pdes = psci.pde.NavierStokes(nu=1 / Re, rho=1.0)

pdes, geo = psci.discretize(pdes, geo, space_nsteps=(50, 50))

bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())

pdes.set_bc_value(bc_value=bc_value)

net = psci.network.FCNet(
    num_ins=2,
    num_outs=3,
    num_layers=10,
    hidden_size=50,
    dtype="float32",
    activation='tanh')

loss = psci.loss.L2(pdes=pdes, geo=geo)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=10000)

rslt = solution(geo)
u = rslt[:, 0]
v = rslt[:, 1]
p = rslt[:, 2]

psci.visu.save_vtk(geo, u, filename="rslt_u")
psci.visu.save_vtk(geo, v, filename="rslt_v")
psci.visu.save_vtk(geo, p, filename="rslt_p")

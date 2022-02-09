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


# Generate BC value
def GenBC(xy, bc_index):
    bc_value = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-4:
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
    return bc_value


# Generate BC weight
def GenBCWeight(xy, bc_index):
    bc_weight = np.zeros((len(bc_index), 2)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - 0.05) < 1e-4:
            bc_weight[i][0] = 1.0 - 20 * abs(xy[id][0])
            bc_weight[i][1] = 1.0
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
    return bc_weight


def GenIns(xy):
    ins = np.zeros((len(xy), 4)).astype(np.float32)
    for i in range(len(xy)):
        ins[i][0] = xy[i][0]
        ins[i][1] = xy[i][1]
        ins[i][2] = 1.0 # TODO
        ins[i][3] = 1.0 # TODO
    return ins


# Time step
dt = 0.1

# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))

# PDE Unstready Navier Stokes
pdes = psci.pde.NavierStokesUnSteady(dt=dt, nu=0.01, rho=1.0)

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(11, 11))


# ins = [x,y,u^n,v^n]
ins = GenIns(geo.get_space_domain())
ins = paddle.to_tensor(ins)

# bc value
bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=4,
    num_outs=3,
    num_layers=10,
    hidden_size=50,
    dtype="float32",
    activation='tanh')

# Loss, TO rename
bc_weight = GenBCWeight(geo.space_domain, geo.bc_index)
loss = psci.loss.L2(pdes=pdes,
                    geo=geo,
                    eq_weight=0.01,
                    bc_weight=bc_weight,
                    synthesis_method='norm')

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())


# Solver
solver = psci.solver.Solver(algo=algo, opt=opt)
solution = solver.solve(num_epoch=30000, ins=ins)

# Use solution
rslt = solution(geo)
u = rslt[:, 0]
v = rslt[:, 1]
u_and_v = np.sqrt(u * u + v * v)
psci.visu.save_vtk(geo, u, filename="rslt_u")
psci.visu.save_vtk(geo, v, filename="rslt_v")
psci.visu.save_vtk(geo, u_and_v, filename="u_and_v")
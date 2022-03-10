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
            bc_weight[i][0] = abs(1.0 - 20 * abs(xy[id][0]))
            bc_weight[i][1] = 1.0
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
    return bc_weight


def GenIns(xy):
    ins = np.zeros((len(xy), 5)).astype(np.float32)
    for i in range(len(xy)):
        ins[i][0] = xy[i][0]
        ins[i][1] = xy[i][1]
        ins[i][2] = 1.0  # TODO
        ins[i][3] = 1.0  # TODO
        ins[i][4] = 1.0  # TODO
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

######

hole = 0.02

sd = geo.get_space_domain()
n = 0
nbc = 0
for i in range(len(sd)):
    x = abs(sd[i][0])
    y = abs(sd[i][1])
    if (x >= hole or y >= hole):  # Space Hole
        n += 1
    if (x == hole or y == hole or abs(x - 0.05) < 1e-4 or
            abs(y - 0.05) < 1e-4):  # Space Hole 
        nbc += 1

sdnew = np.ndarray((n, 2), dtype='float32')
bcidxnew = np.ndarray(nbc, dtype='int64')

# print(n)
# print(nbc)

n = 0
for i in range(len(sd)):
    x = abs(sd[i][0])
    y = abs(sd[i][1])
    if (x >= hole or y >= hole):  # Space Hole
        sdnew[n][0] = sd[i][0]
        sdnew[n][1] = sd[i][1]
        n += 1

nbc = 0
for i in range(len(sdnew)):
    x = abs(sdnew[i][0])
    y = abs(sdnew[i][1])
    if (abs(x) == hole or abs(y) == hole or abs(x - 0.05) < 1e-4 or
            abs(y - 0.05) < 1e-4):  # Space Hole
        bcidxnew[nbc] = i
        nbc += 1

# geo.space_domain = sdnew
# geo.bc_index = bcidxnew

# print(geo.space_domain)
# print(sdnew.shape)

# print(geo.bc_index)

# print(sdnew)
# print(bcidxnew)

##### 

# ins = [x,y,u^n,v^n,p^n]
ins = GenIns(geo.get_space_domain())
ins = paddle.to_tensor(ins, stop_gradient=False, dtype="float32")

# bc value
bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=5,
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
rslt = solution(ins)
u = rslt[:, 0]
v = rslt[:, 1]
u_and_v = np.sqrt(u * u + v * v)
psci.visu.save_vtk(geo, u, filename="rslt_u")
psci.visu.save_vtk(geo, v, filename="rslt_v")
psci.visu.save_vtk(geo, u_and_v, filename="u_and_v")

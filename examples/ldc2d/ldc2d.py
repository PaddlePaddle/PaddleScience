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


# Geometry
geo = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))

# PDE Laplace
pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0)

# Discretization
pdes, geo = psci.discretize(pdes, geo, space_nsteps=(101, 101))

# bc value
bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())
pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1])

# Network
net = psci.network.FCNet(
    num_ins=2,
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
solution = solver.solve(num_epoch=30000)

# Use solution
rslt = solution(geo)
u = rslt[:, 0]
v = rslt[:, 1]
u_and_v = np.sqrt(u * u + v * v)
psci.visu.save_vtk(geo, u, filename="rslt_u")
psci.visu.save_vtk(geo, v, filename="rslt_v")
psci.visu.save_vtk(geo, u_and_v, filename="u_and_v")

openfoam_u = np.load("./openfoam/openfoam_u_100.npy")
diff_u = u - openfoam_u
RSE_u = np.linalg.norm(diff_u, ord=2)
MSE_u = RSE_u * RSE_u / geo.get_domain_size()
print("MSE_u: ", MSE_u)
openfoam_v = np.load("./openfoam/openfoam_v_100.npy")
diff_v = v - openfoam_v
RSE_v = np.linalg.norm(diff_v, ord=2)
MSE_v = RSE_v * RSE_v / geo.get_domain_size()
print("MSE_v: ", MSE_v)

# Infer with another geometry
geo_1 = psci.geometry.Rectangular(
    space_origin=(-0.05, -0.05), space_extent=(0.05, 0.05))
geo_1 = geo_1.discretize(space_nsteps=(401, 401))
rslt_1 = solution(geo_1)
u_1 = rslt_1[:, 0]
v_1 = rslt_1[:, 1]
psci.visu.save_vtk(geo_1, u_1, filename="rslt_u_400")
psci.visu.save_vtk(geo_1, v_1, filename="rslt_v_400")

openfoam_u_1 = np.load("./openfoam/openfoam_u_400.npy")
diff_u_1 = u_1 - openfoam_u_1
RSE_u_1 = np.linalg.norm(diff_u_1, ord=2)
MSE_u_1 = RSE_u_1 * RSE_u_1 / geo_1.get_domain_size()
print("MSE_u_400: ", MSE_u_1)
openfoam_v_1 = np.load("./openfoam/openfoam_v_400.npy")
diff_v_1 = v_1 - openfoam_v_1
RSE_v_1 = np.linalg.norm(diff_v_1, ord=2)
MSE_v_1 = RSE_v_1 * RSE_v_1 / geo_1.get_domain_size()
print("MSE_v_400: ", MSE_v_1)

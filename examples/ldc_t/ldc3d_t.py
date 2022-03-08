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


# Generate BC value
# Every row have 3 elements which means u,v,w of one point in one moment
def GenBC(txyz, bc_index):
    bc_value = np.zeros((len(bc_index), 3)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        # if the point is in top surface
        if abs(txyz[id][3] - 0.05) < 1e-4:
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
            bc_value[i][2] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
            bc_value[i][2] = 0.0
    return bc_value


# Generate IC value
def GenIC(txyz, ic_index):
    ic_value = np.zeros((len(ic_index), 3)).astype(np.float32)
    for i in range(len(ic_index)):
        id = ic_index[i]
        if abs(txyz[id][3] - 0.05) < 1e-4:
            ic_value[i][0] = 1.0
            ic_value[i][1] = 0.0
            ic_value[i][2] = 0.0
        else:
            ic_value[i][0] = 0.0
            ic_value[i][1] = 0.0
            ic_value[i][2] = 0.0
    return ic_value


# Get a section of the 3D geo
def GetSection(xyz, y=0.01):
    section_index = []
    for i in range(len(xyz)):
        if (np.abs(xyz[i][2] - y)) < 1e-4:
            section_index.append(i)
    return np.array(section_index)


if __name__ == "__main__":
    # Geometry
    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=0.5,
        space_origin=(-0.05, 0, -0.05),
        space_extent=(0.05, 0.05, 0.05))

    # PDE Laplace
    pdes = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=3, time_dependent=True)

    # Discretization
    pdes, geo = psci.sampling_discretize(
        pdes,
        geo,
        time_nsteps=6,
        space_point_size=1000,
        space_nsteps=(21, 6, 21))

    # bc value
    bc_value = GenBC(geo.get_domain(), geo.get_bc_index())
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1, 2])

    # ic value
    ic_value = GenIC(geo.get_domain(), geo.get_ic_index())
    pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1, 2])

    # Network
    net = psci.network.FCNet(
        num_ins=4,
        num_outs=4,
        num_layers=10,
        hidden_size=50,
        dtype="float32",
        activation='tanh')

    # Loss, TO rename
    loss = psci.loss.L2(pdes=pdes,
                        geo=geo,
                        eq_weight=0.01,
                        synthesis_method='norm')

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    # Solver
    solver = psci.solver.Solver(algo=algo, opt=opt)
    solution = solver.solve(num_epoch=10)

    # Use solution
    rslt = solution(geo)
    # Get the result of last moment
    rslt = rslt[(-geo.space_domain_size):, :]

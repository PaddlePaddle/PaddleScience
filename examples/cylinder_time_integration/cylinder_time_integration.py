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
# Every row have 4 elements which means u,v,w,p of one point in one moment
def GenBC(xyz, bc_index):
    bc_value = np.zeros((len(bc_index), 4)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        # if the point is in top surface 
        # if the point is in cylinder
        if (abs(xyz[id][2] - 0.05) < 1e-4):
            bc_value[i][0] = 1.0
            bc_value[i][1] = 0.0
            bc_value[i][2] = 0.0
            bc_value[i][3] = 0.0
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
            bc_value[i][2] = 0.0
            bc_value[i][3] = 0.0
    return bc_value


def GenInitPhyInfo(xyz):
    uvwp = np.zeros((len(xyz), 4)).astype(np.float32)
    return uvwp


# Generate BC weight
# TODO(Liuxiandong)
def GenBCWeight(xyz, bc_index):
    bc_weight = np.zeros((len(bc_index), 3)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xyz[id][1] - 0.05) < 1e-4:
            bc_weight[i][0] = 1.0 - 20 * abs(xyz[id][0])
            bc_weight[i][1] = 1.0
            bc_weight[i][2] = 1.0
        else:
            bc_weight[i][0] = 1.0
            bc_weight[i][1] = 1.0
            bc_weight[i][2] = 1.0
    return bc_weight


if __name__ == "__main__":
    # Geometry
    geo = psci.geometry.CylinderInRectangular(
        space_origin=(-0.05, -0.05, -0.05),
        space_extent=(0.05, 0.05, 0.05),
        circle_center=(0, 0),
        circle_radius=0.02)

    # PDE Laplace
    pdes = psci.pde.NavierStokes(
        nu=0.01, rho=1.0, dim=3, time_integration=True, dt=0.1)

    # Discretization
    geo = psci.geometry.CylinderInRectangular.sampling_discretize(
        geo, space_npoints=1000, space_nsteps=(31, 31, 31), circle_bc_size=100)

    # bc value
    bc_value = GenBC(geo.get_space_domain(), geo.get_bc_index())
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1, 2, 3])

    # uvwp value of t0 time
    uvwp = GenInitPhyInfo(geo.get_space_domain())

    # Network
    net = psci.network.FCNet(
        num_ins=7,
        num_outs=4,
        num_layers=10,
        hidden_size=50,
        dtype="float32",
        activation='tanh')

    # Loss, TO rename 
    loss = psci.loss.L2(pdes=pdes,
                        geo=geo,
                        physic_info=uvwp,
                        eq_weight=0.01,
                        synthesis_method='norm')

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    # Solver
    solver = psci.solver.Solver(algo=algo, opt=opt)
    solution = solver.solve(num_epoch=100)

    # Use solution for inference
    time_nsteps = 5
    check_time = 2
    current_physic_info = uvwp
    for i in range(time_nsteps):
        print(" inference for %d time step" % (i + 1))
        rslt = solution(geo, physic_info=current_physic_info)
        current_physic_info = rslt
        if i + 1 == check_time:
            u = rslt[:, 0]
            v = rslt[:, 1]
            w = rslt[:, 2]
            p = rslt[:, 3]
            # output the result
            rslt_dictionary = {'u': u, 'v': v, 'w': w, 'p': p}
            psci.visu.save_vtk_points(
                filename="FlowAroundCyclinder", geo=geo, data=rslt_dictionary)

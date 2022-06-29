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

import loading_cfd_data
import numpy as np

paddle.seed(1)
np.random.seed(1)

# time array
time_tmp = np.linspace(0, 50, 50, endpoint=True).astype(int)
time_array = np.random.choice(time_tmp, 11)
time_array.sort()

time_array = np.array([0, 1])

# loading data from files
dr = loading_cfd_data.DataLoader(path='./datasets/')

# boundary inlet and circle
b_inlet_u, b_inlet_v, t, b_inlet_x, b_inlet_y = dr.loading_boundary_data([1])

# boundary outlet
b_outlet_p, t, b_outlet_x, b_outlet_y = dr.loading_outlet_data([1])

# initial data
init_p, init_u, init_v, t, init_x, init_y = dr.loading_initial_data([1])
init_x = init_x.astype('float32')
init_y = init_y.astype('float32')

# supervised data
sup_p, sup_u, sup_v, t, sup_x, sup_y = dr.loading_supervised_data(time_array[
    1::])

# discrete geometry
geo_disc = psci.geometry.GeometryDiscrete()
geo_disc.interior = np.stack((init_x.flatten(), init_y.flatten()), axis=1)
geo_disc.boundary["inlet"] = np.stack(
    (b_inlet_x.flatten(), b_inlet_y.flatten()), axis=1)
geo_disc.boundary["outlet"] = np.stack(
    (b_outlet_x.flatten(), b_outlet_y.flatten()), axis=1)
n1 = int(sup_x.shape[0] / (len(time_array) - 1))
geo_disc.user = np.stack(
    (sup_x[0:n1].flatten(), sup_y[0:n1].flatten()), axis=1)

# N-S
pde = psci.pde.NavierStokes(nu=0.02, rho=1.0, dim=2, time_dependent=True)

# set bounday condition
bc_inlet_u = psci.bc.Dirichlet('u', rhs=b_inlet_u.flatten(), weight=10.0)
bc_inlet_v = psci.bc.Dirichlet('v', rhs=b_inlet_v.flatten(), weight=10.0)
bc_outlet_p = psci.bc.Dirichlet('p', rhs=b_outlet_p.flatten(), weight=1.0)

# add bounday and boundary condition
pde.add_bc("inlet", bc_inlet_u, bc_inlet_v)
pde.add_bc("outlet", bc_outlet_p)

# add initial condition
ic_u = psci.ic.IC('u', rhs=init_u.flatten(), weight=10.0)
ic_v = psci.ic.IC('v', rhs=init_v.flatten(), weight=10.0)
ic_p = psci.ic.IC('p', rhs=init_p.flatten(), weight=10.0)
pde.add_ic(ic_u, ic_v, ic_p)

# discretization pde
pde_disc = pde.discretize(time_array=time_array, geo_disc=geo_disc)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=3, num_layers=6, hidden_size=50, activation='tanh')

net_params = './checkpoint/pretrained_net_params'
load_params = paddle.load(net_params)
net.set_state_dict(load_params)

# Loss
loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)

# Set supervised data
sup_data = np.stack(
    (sup_p.flatten(), sup_u.flatten(), sup_v.flatten()), axis=1)
solver.feed_data_user(sup_data)

solution = solver.solve(num_epoch=10)

psci.visu.save_vtk(
    time_array=pde_disc.time_array, geo_disc=pde_disc.geometry, data=solution)

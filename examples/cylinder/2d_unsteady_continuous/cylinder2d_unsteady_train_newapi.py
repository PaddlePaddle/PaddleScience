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
"""This *.py file is an example of solving 2d-unsteady-cylinder–flow, by using PINNs method."""
import os
import numpy as np
import paddlescience as psci
import paddle
import loading_cfd_data

paddle.seed(1)
np.random.seed(1)

# paddle.enable_static()f
dirname =  os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(dirname)
# time array
time_tmp = np.linspace(0, 50, 50, endpoint=True).astype(int)
time_array = np.random.choice(time_tmp, 11)
time_array.sort()

time_array = np.array([1, 2, 3])

# loading data from files
dr = loading_cfd_data.DataLoader(path='./datasets/')
# interior data
i_t, i_x, i_y = dr.loading_train_inside_domain_data(
    time_array, flatten=True, dtype='float32')
# boundary inlet and circle
b_inlet_u, b_inlet_v, b_inlet_t, b_inlet_x, b_inlet_y = dr.loading_boundary_data(
    time_array, flatten=True, dtype='float32')
# boundary outlet
b_outlet_p, b_outlet_t, b_outlet_x, b_outlet_y = dr.loading_outlet_data(
    time_array, flatten=True, dtype='float32')
# initial data
init_p, init_u, init_v, init_t, init_x, init_y = dr.loading_initial_data(
    [1], flatten=True, dtype='float32')
# supervised data
sup_p, sup_u, sup_v, sup_t, sup_x, sup_y = dr.loading_supervised_data(
    time_array, flatten=True, dtype='float32')

inputeq = np.stack((i_t, i_x, i_y), axis=1)
inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y), axis=1)
inputbc2 = np.stack((b_outlet_t, b_outlet_x, b_outlet_y), axis=1)
inputic = np.stack((init_t, init_x, init_y), axis=1)
inputsup = np.stack((sup_t, sup_x, sup_y), axis=1)
refsup = np.stack((sup_p, sup_u, sup_v), axis=1)

# N-S
pde = psci.pde.NavierStokes(nu=0.02, rho=1.0, dim=2, time_dependent=True)

# set bounday condition
bc_inlet_u = psci.bc.Dirichlet('u', rhs=b_inlet_u)
bc_inlet_v = psci.bc.Dirichlet('v', rhs=b_inlet_v)
bc_outlet_p = psci.bc.Dirichlet('p', rhs=b_outlet_p)

# add bounday and boundary condition
pde.set_bc("inlet", bc_inlet_u, bc_inlet_v)
pde.set_bc("outlet", bc_outlet_p)

# add initial condition
ic_u = psci.ic.IC('u', rhs=init_u)
ic_v = psci.ic.IC('v', rhs=init_v)
ic_p = psci.ic.IC('p', rhs=init_p)
pde.set_ic(ic_u, ic_v, ic_p)

# Network
net = psci.network.FCNet(
    num_ins=3, num_outs=3, num_layers=6, hidden_size=50, activation='tanh')
net.initialize(path='./checkpoint/pretrained_net_params')

outeq = net(inputeq)
outbc1 = net(inputbc1)
outbc2 = net(inputbc2)
outic = net(inputic)
outsup = net(inputsup)

# eq loss
losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
# bc loss
lossbc1 = psci.loss.BcLoss("inlet", netout=outbc1)
lossbc2 = psci.loss.BcLoss("outlet", netout=outbc2)
# ic loss
lossic = psci.loss.IcLoss(netout=outic)
# supervise loss
losssup = psci.loss.DataLoss(netout=outsup, ref=refsup)

# total loss
loss = losseq1 + losseq2 + losseq3 + 10.0 * lossbc1 + lossbc2 + 10.0 * lossic + 10.0 * losssup

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)

# Solve
solution = solver.solve(num_epoch=20)

# Save last time data to vtk
n = int(i_x.shape[0] / len(time_array))
cord = np.stack(
    (i_x[0:n].astype("float32"), i_y[0:n].astype("float32")), axis=1)
psci.visu.__save_vtk_raw(cordinate=cord, data=solution[0][-n::])

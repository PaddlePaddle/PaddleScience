# # Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import copy

# import numpy as np
# import paddle
# import paddle.distributed as dist
# import paddlescience as psci
# from pyevtk.hl import pointsToVTK

# import sample_boundary_training_data as sample_data
# import vtk
# from load_lbm_data import load_ic_data, load_supervised_data

# paddle.seed(42)
# np.random.seed(42)
# # paddle.enable_static()

# # time arraep
# t_start = 2000.1
# # t_end = 2002.1
# t_end = 2000.5
# t_step = 0.1
# time_num = int((t_end - t_start + t_step)/t_step)
# time_array = np.linspace(t_start - 2000, t_end - 2000, time_num, endpoint=True)
# # time_array = np.random.choice(time_array, time_num)
# # time_array.sort()
# print(f"time_num = {time_num}, time_array = {time_array}")

# ic_t = 2000.0
# n_sup = 2000
# txyz_uvwpe_ic = load_ic_data(ic_t)
# txyz_uvwpe_s = load_supervised_data(t_start, t_end, t_step, ic_t, n_sup)

# # num points to sample per GPU
# num_points = 15000
# # discretize node by geo
# inlet_txyz, outlet_txyz, cylinder_txyz, interior_txyz = sample_data.sample_data(t_step=time_num, nr_points=num_points)

# i_t = interior_txyz[:, 0]
# i_x = interior_txyz[:, 1]
# i_y = interior_txyz[:, 2]
# i_z = interior_txyz[:, 3]

# # initial value
# init_t = txyz_uvwpe_ic[:, 0]
# init_x = txyz_uvwpe_ic[:, 1]
# init_y = txyz_uvwpe_ic[:, 2]
# init_z = txyz_uvwpe_ic[:, 3]
# init_u = txyz_uvwpe_ic[:, 4]
# init_v = txyz_uvwpe_ic[:, 5]
# init_w = txyz_uvwpe_ic[:, 6]
# init_p = txyz_uvwpe_ic[:, 7]

# # supervised data
# sup_t = txyz_uvwpe_s[:, 0]
# sup_x = txyz_uvwpe_s[:, 1]
# sup_y = txyz_uvwpe_s[:, 2]
# sup_z = txyz_uvwpe_s[:, 3]
# sup_u = txyz_uvwpe_s[:, 4]
# sup_v = txyz_uvwpe_s[:, 5]
# sup_w = txyz_uvwpe_s[:, 6]
# sup_p = txyz_uvwpe_s[:, 7]

# # bc inlet nodes discre
# b_inlet_t = inlet_txyz[:, 0]
# b_inlet_x = inlet_txyz[:, 1]
# b_inlet_y = inlet_txyz[:, 2]
# b_inlet_z = inlet_txyz[:, 3]

# # bc outlet nodes discre
# b_outlet_t = outlet_txyz[:, 0]
# b_outlet_x = outlet_txyz[:, 1]
# b_outlet_y = outlet_txyz[:, 2]
# b_outlet_z = outlet_txyz[:, 3]

# # bc cylinder nodes discre
# b_cylinder_t = cylinder_txyz[:, 0]
# b_cylinder_x = cylinder_txyz[:, 1]
# b_cylinder_y = cylinder_txyz[:, 2]
# b_cylinder_z = cylinder_txyz[:, 3]

# # bc & interior nodes for nn
# inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)
# inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z), axis=1)
# inputbc2 = np.stack((b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z), axis=1)
# inputbc3 = np.stack((b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z), axis=1)
# inputic = np.stack((init_t, init_x, init_y, init_z), axis=1)
# inputsup = np.stack((sup_t, sup_x, sup_y, sup_z), axis=1)
# refsup = np.stack((sup_u, sup_v, sup_w), axis=1)

# # N-S, Re=3900, D=80, u=1, nu=8/390
# pde = psci.pde.NavierStokes(nu=0.0205, rho=1.0, dim=3, time_dependent=True)

# # set bounday condition
# bc_inlet_u = psci.bc.Dirichlet("u", rhs=1)
# bc_inlet_v = psci.bc.Dirichlet("v", rhs=0)
# bc_inlet_w = psci.bc.Dirichlet("w", rhs=0)
# bc_cylinder_u = psci.bc.Dirichlet("u", rhs=0)
# bc_cylinder_v = psci.bc.Dirichlet("v", rhs=0)
# bc_cylinder_w = psci.bc.Dirichlet("w", rhs=0)
# bc_outlet_p = psci.bc.Dirichlet("p", rhs=0)

# # add bounday and boundary condition
# pde.set_bc("inlet", bc_inlet_u, bc_inlet_v, bc_inlet_w)
# pde.set_bc("cylinder", bc_cylinder_u, bc_cylinder_v, bc_cylinder_w)
# pde.set_bc("outlet", bc_outlet_p)

# # add initial condition
# ic_u = psci.ic.IC("u", rhs=init_u)
# ic_v = psci.ic.IC("v", rhs=init_v)
# ic_w = psci.ic.IC("w", rhs=init_w)
# ic_p = psci.ic.IC("p", rhs=init_p)
# pde.set_ic(ic_u, ic_v, ic_w, ic_p)

# # Network
# net = psci.network.FCNet(
#     num_ins=4, num_outs=4, num_layers=6, hidden_size=50, activation="tanh")
# # net.initialize("checkpoint/static_model_params_10000.pdparams")

# outeq = net(inputeq)
# outbc1 = net(inputbc1)
# outbc2 = net(inputbc2)
# outbc3 = net(inputbc3)
# outic = net(inputic)
# outsup = net(inputsup)

# # eq loss
# losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
# losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
# losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
# losseq4 = psci.loss.EqLoss(pde.equations[3], netout=outeq)

# # bc loss
# lossbc1 = psci.loss.BcLoss("inlet", netout=outbc1)
# lossbc2 = psci.loss.BcLoss("outlet", netout=outbc2)
# lossbc3 = psci.loss.BcLoss("cylinder", netout=outbc3)

# # ic loss
# lossic = psci.loss.IcLoss(netout=outic)

# # supervise loss
# losssup = psci.loss.DataLoss(netout=outsup[0:3], ref=refsup)

# # total loss
# loss = losseq1 + losseq2 + losseq3 + losseq4 + 10.0 * lossbc1 + lossbc2 + lossbc3 + 10.0 * lossic + losssup

# # Algorithm
# algo = psci.algorithm.PINNs(net=net, loss=loss)

# # Optimizer
# opt = psci.optimizer.Adam(learning_rate=0.01, parameters=net.parameters())

# # Solver
# solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)

# # Solve
# solution = solver.solve(num_epoch=100000)

# for i in solution:
#     print(i.shape)

# n = int(i_x.shape[0] / len(time_array))
# i_x = i_x.astype("float32")
# i_y = i_y.astype("float32")
# i_z = i_z.astype("float32")

# cord = np.stack((i_x[0:n], i_y[0:n], i_z[0:n]), axis=1)
# # psci.visu.__save_vtk_raw(cordinate=cord, data=solution[0][-n::])
# psci.visu.save_vtk_cord(filename="output", time_array=time_array, cord=cord, data=solution)

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

import copy

import numpy as np
import paddle
import paddle.distributed as dist
import paddlescience as psci
from pyevtk.hl import pointsToVTK

import sample_boundary_training_data as sample_data
import vtk
from load_lbm_data import load_ic_data, load_supervised_data

paddle.seed(42)
np.random.seed(42)
# paddle.enable_static()

# time arraep
ic_t = 200000
t_start = 200050
t_end = 200250
t_step = 50
time_num = int((t_end - t_start) / t_step) + 1
time_tmp = np.linspace(t_start - ic_t, t_end - ic_t, time_num, endpoint=True)
# time_array = np.random.choice(time_tmp, time_num, replace=False)
time_array = time_tmp
time_array.sort()
print(f"time_num = {time_num}, time_array = {time_array}")

# weight of losses - inlet, outlet, cylinder, top wall, bottom wall, equation, initial condition, supervised data
inlet_wgt = 2.0
outlet_wgt = 1.0
cylinder_wgt = 5.0
top_wgt = 2.0
bottom_wgt = 2.0
eq_wgt= 10.0
ic_wgt = 5.0
sup_wgt = 10.0

# initial value
txyz_uvwpe_ic = load_ic_data(ic_t)
init_t = txyz_uvwpe_ic[:, 0]
init_x = txyz_uvwpe_ic[:, 1]
init_y = txyz_uvwpe_ic[:, 2]
init_z = txyz_uvwpe_ic[:, 3]
init_u = txyz_uvwpe_ic[:, 4]
init_v = txyz_uvwpe_ic[:, 5]
init_w = txyz_uvwpe_ic[:, 6]
init_p = txyz_uvwpe_ic[:, 7]

# num of supervised points
n_sup = 2000

# supervised data
txyz_uvwpe_s = load_supervised_data(t_start, t_end, t_step, ic_t, n_sup)
sup_t = txyz_uvwpe_s[:, 0]
sup_x = txyz_uvwpe_s[:, 1]
sup_y = txyz_uvwpe_s[:, 2]
sup_z = txyz_uvwpe_s[:, 3]
sup_u = txyz_uvwpe_s[:, 4]
sup_v = txyz_uvwpe_s[:, 5]
sup_w = txyz_uvwpe_s[:, 6]
sup_p = txyz_uvwpe_s[:, 7]

# num points to sample per GPU
# num_points = 30000
num_points = 15000
# discretize node by geo
inlet_txyz, outlet_txyz, top_txyz, bottom_txyz, cylinder_txyz, interior_txyz = sample_data.sample_data(t_step=time_num, nr_points=num_points)

# interior nodes discre
i_t = interior_txyz[:, 0]
i_x = interior_txyz[:, 1]
i_y = interior_txyz[:, 2]
i_z = interior_txyz[:, 3]

# bc inlet nodes discre
b_inlet_t = inlet_txyz[:, 0]
b_inlet_x = inlet_txyz[:, 1]
b_inlet_y = inlet_txyz[:, 2]
b_inlet_z = inlet_txyz[:, 3]

# bc outlet nodes discre
b_outlet_t = outlet_txyz[:, 0]
b_outlet_x = outlet_txyz[:, 1]
b_outlet_y = outlet_txyz[:, 2]
b_outlet_z = outlet_txyz[:, 3]

# bc cylinder nodes discre
b_cylinder_t = cylinder_txyz[:, 0]
b_cylinder_x = cylinder_txyz[:, 1]
b_cylinder_y = cylinder_txyz[:, 2]
b_cylinder_z = cylinder_txyz[:, 3]

# bc-top nodes discre
b_top_t = top_txyz[:, 0] # value = [1, 2, 3, 4, 5]
b_top_x = top_txyz[:, 1]
b_top_y = top_txyz[:, 2]
b_top_z = top_txyz[:, 3]
 
# bc-bottom nodes discre
b_bottom_t = bottom_txyz[:, 0] # value = [1, 2, 3, 4, 5]
b_bottom_x = bottom_txyz[:, 1]
b_bottom_y = bottom_txyz[:, 2]
b_bottom_z = bottom_txyz[:, 3]

# bc & interior nodes for nn
inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)
inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z), axis=1)
inputbc2 = np.stack((b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z), axis=1)
inputbc3 = np.stack((b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z), axis=1)
inputbc4_top = np.stack((b_top_t, b_top_x, b_top_y, b_top_z), axis=1)
inputbc5_bottom = np.stack((b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z), axis=1)

inputic = np.stack((init_t, init_x, init_y, init_z), axis=1)
inputsup = np.stack((sup_t, sup_x, sup_y, sup_z), axis=1)
refsup = np.stack((sup_u, sup_v, sup_w), axis=1)

# N-S, Re=3900, D=80, u=1, nu=8/390
pde = psci.pde.NavierStokes(nu=0.0205, rho=1.0, dim=3, time_dependent=True)

# set bounday condition
bc_inlet_u = psci.bc.Dirichlet("u", rhs=1)
bc_inlet_v = psci.bc.Dirichlet("v", rhs=0)
bc_inlet_w = psci.bc.Dirichlet("w", rhs=0)
bc_cylinder_u = psci.bc.Dirichlet("u", rhs=0)
bc_cylinder_v = psci.bc.Dirichlet("v", rhs=0)
bc_cylinder_w = psci.bc.Dirichlet("w", rhs=0)
bc_outlet_p = psci.bc.Dirichlet("p", rhs=0)
bc_top_u = psci.bc.Dirichlet("u", rhs=0)
bc_top_v = psci.bc.Dirichlet("v", rhs=0)
bc_bottom_u = psci.bc.Dirichlet("u", rhs=0)
bc_bottom_v = psci.bc.Dirichlet("v", rhs=0)

# add bounday and boundary condition
pde.set_bc("inlet", bc_inlet_u, bc_inlet_v, bc_inlet_w)
pde.set_bc("cylinder", bc_cylinder_u, bc_cylinder_v, bc_cylinder_w)
pde.set_bc("outlet", bc_outlet_p)
pde.set_bc("top", bc_top_u, bc_top_v)
pde.set_bc("bottom", bc_bottom_u, bc_bottom_v)

# add initial condition
ic_u = psci.ic.IC("u", rhs=init_u)
ic_v = psci.ic.IC("v", rhs=init_v)
ic_w = psci.ic.IC("w", rhs=init_w)
ic_p = psci.ic.IC("p", rhs=init_p)
pde.set_ic(ic_u, ic_v, ic_w)

# Network
net = psci.network.FCNet(
    num_ins=4, num_outs=4, num_layers=6, hidden_size=50, activation="tanh")
# net.initialize("checkpoint/static_model_params_10000.pdparams")
# net.initialize("checkpoint/dynamic_net_params_100000.pdparams")

outeq = net(inputeq)
outbc1 = net(inputbc1)
outbc2 = net(inputbc2)
outbc3 = net(inputbc3)
outbc4 = net(inputbc4_top)
outbc5 = net(inputbc5_bottom)
outic = net(inputic)
outsup = net(inputsup)

# eq loss
losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
losseq4 = psci.loss.EqLoss(pde.equations[3], netout=outeq)

# bc loss
lossbc1 = psci.loss.BcLoss("inlet", netout=outbc1)
lossbc2 = psci.loss.BcLoss("outlet", netout=outbc2)
lossbc3 = psci.loss.BcLoss("cylinder", netout=outbc3)
lossbc4 = psci.loss.BcLoss("top", netout=outbc4)
lossbc5 = psci.loss.BcLoss("bottom", netout=outbc5)

# ic loss
lossic = psci.loss.IcLoss(netout=outic)

# supervise loss
losssup = psci.loss.DataLoss(netout=outsup[0:3], ref=refsup)

# total loss
loss = losseq1 * eq_wgt+ losseq2 * eq_wgt + losseq3 * eq_wgt + losseq4 * eq_wgt + \
    lossbc1 * inlet_wgt + \
    lossbc2 * outlet_wgt + \
    lossbc3 * cylinder_wgt + \
    lossbc4 * top_wgt + \
    lossbc5 * bottom_wgt + \
    lossic * ic_wgt + \
    losssup * sup_wgt
# loss = lossic

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.01, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)

# Solve
solution = solver.solve(num_epoch=100000)
# solution = solver.predict()
# print(type(solution), len(solution), solution[0].shape)
# exit()

# solution = [
#     np.stack([init_u, init_v, init_w, init_p], axis=1)
# ]
for idx, si in enumerate(solution):
    print(f"solution[{idx}].shape = {si.shape}")

n = int(i_x.shape[0] / len(time_array))
# n = 1
i_x = i_x.astype("float32")
i_y = i_y.astype("float32")
i_z = i_z.astype("float32")

# denormalize back
i_x = i_x * 1600
i_y = i_y * 800
i_z = i_z * 320

cord = np.stack((i_x[0:n], i_y[0:n], i_z[0:n]), axis=1)
# cord = np.stack((i_x, i_y, i_z), axis=1)
# psci.visu.__save_vtk_raw(cordinate=cord, data=solution[0][-n::])
psci.visu.save_vtk_cord(filename="./vtk/output_2023_1_12", time_array=time_array, cord=cord, data=solution)

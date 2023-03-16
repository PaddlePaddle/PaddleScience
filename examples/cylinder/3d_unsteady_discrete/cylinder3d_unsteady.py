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


import os
import sys
import paddle
import numpy as np
import paddlescience as psci
import sample_boundary_training_data as sample_data
from load_lbm_data import load_vtk


Re = 3900
U0 = 0.1
Dcylinder = 80.0
rho = 1.0
nu = rho * U0 * Dcylinder / Re

t_star = Dcylinder / U0 # 800
xyz_star = Dcylinder    # 80
uvw_star = U0           # 0.1
p_star = rho * U0 * U0  # 0.01

# read configuration file : config.yaml
dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
config = psci.utils.get_config(fname=dirname + r'/config.yaml', config_index="hyper parameters")
if config is not None:
    # number of epoch
    num_epoch = config['number of epoch']
    learning_rate = config['learning rate']
    hidden_size = config["hidden size"]
    # batch size
    bs = {}
    bs['interior'] = config['batch size']['interior']

    bs['inlet'] = config['batch size']['inlet']
    bs['outlet'] = config['batch size']['outlet']
    bs['cylinder'] = config['batch size']['cylinder']
    bs['top'] = config['batch size']['top']
    bs['bottom'] = config['batch size']['bottom']
    
    bs['ic'] = config['batch size']['initial condition']
    bs['supervised'] = config['batch size']['supervised']
    
    # losses weight
    ic_wgt = config['weight of losses']['initial condition']
    eq_wgt= config['weight of losses']['pde']

    front_wgt = config['weight of losses']['front']
    back_wgt = config['weight of losses']['back']
    inlet_wgt = config['weight of losses']['left inlet']
    outlet_wgt = config['weight of losses']['right outlet']
    top_wgt = config['weight of losses']['top']
    bottom_wgt = config['weight of losses']['bottom']
    cylinder_wgt = config['weight of losses']['cylinder']

    sup_wgt = config['weight of losses']['supervised']

    # simulated annealing
    w_epoch = config['Simulated Annealing']['warm epochs']
    
    # epoch number
    seed_number = config["random seed"]
else:
    print("Error : mssing configure files !")

use_random_time = False

# fix the random seed
paddle.seed(seed_number)
np.random.seed(seed_number)

# time array
ic_t = 200000
t_start = 200050
t_end = 204950
t_step = 50
time_num = int((t_end - t_start) / t_step) + 1
time_list = np.linspace(int((t_start - ic_t) / t_step), int((t_end - ic_t) / t_step), time_num, endpoint=True).astype(int)
# time_tmp = np.linspace(t_start - ic_t, t_end - ic_t, time_num, endpoint=True)
time_tmp = time_list * t_step
time_index = np.random.choice(time_list, int(time_num / 2.5), replace=False)
time_index.sort()
time_array = time_index * t_step
print(f"time_num = {time_num}")
print(f"time_list = {time_list}")
print(f"time_tmp = {time_tmp}")
print(f"time_index = {time_index}")
print(f"time_array = {time_array}")

# initial value
ic_name = dirname + r'/data/LBM_result/cylinder3d_2023_1_31_LBM_'
txyz_uvwpe_ic = load_vtk([0], t_step=50, load_uvwp=True, load_txyz=True, name_wt_time=ic_name)[0]
init_t = txyz_uvwpe_ic[:, 0] / t_star; print(f"init_t={init_t.shape} {init_t.mean().item():.10f}")
init_x = txyz_uvwpe_ic[:, 1] / xyz_star; print(f"init_x={init_x.shape} {init_x.mean().item():.10f}")
init_y = txyz_uvwpe_ic[:, 2] / xyz_star; print(f"init_y={init_y.shape} {init_y.mean().item():.10f}")
init_z = txyz_uvwpe_ic[:, 3] / xyz_star; print(f"init_z={init_z.shape} {init_z.mean().item():.10f}")
init_u = txyz_uvwpe_ic[:, 4] / uvw_star; print(f"init_u={init_u.shape} {init_u.mean().item():.10f}")
init_v = txyz_uvwpe_ic[:, 5] / uvw_star; print(f"init_v={init_v.shape} {init_v.mean().item():.10f}")
init_w = txyz_uvwpe_ic[:, 6] / uvw_star; print(f"init_w={init_w.shape} {init_w.mean().item():.10f}")
init_p = txyz_uvwpe_ic[:, 7] / p_star; print(f"init_p={init_p.shape} {init_p.mean().item():.10f}")

# num of supervised points
n_sup = 2000

# supervised data
sup_name = dirname + r"/data/sup_data/supervised_"
txyz_uvwpe_s_new = load_vtk(time_index, t_step=t_step, load_uvwp=True, load_txyz=True, name_wt_time=sup_name)
txyz_uvwpe_s = np.zeros((0,8))
for i in range(len(txyz_uvwpe_s_new)):
    txyz_uvwpe_s = np.concatenate((txyz_uvwpe_s, txyz_uvwpe_s_new[i][:, :]), axis=0)
sup_t = txyz_uvwpe_s[:, 0] / t_star; print(f"sup_t={sup_t.shape} {sup_t.mean().item():.10f}")
sup_x = txyz_uvwpe_s[:, 1] / xyz_star; print(f"sup_x={sup_x.shape} {sup_x.mean().item():.10f}")
sup_y = txyz_uvwpe_s[:, 2] / xyz_star; print(f"sup_y={sup_y.shape} {sup_y.mean().item():.10f}")
sup_z = txyz_uvwpe_s[:, 3] / xyz_star; print(f"sup_z={sup_z.shape} {sup_z.mean().item():.10f}")
sup_u = txyz_uvwpe_s[:, 4] / uvw_star; print(f"sup_u={sup_u.shape} {sup_u.mean().item():.10f}")
sup_v = txyz_uvwpe_s[:, 5] / uvw_star; print(f"sup_v={sup_v.shape} {sup_v.mean().item():.10f}")
sup_w = txyz_uvwpe_s[:, 6] / uvw_star; print(f"sup_w={sup_w.shape} {sup_w.mean().item():.10f}")
sup_p = txyz_uvwpe_s[:, 7] / p_star; print(f"sup_p={sup_p.shape} {sup_p.mean().item():.10f}")

# num of interior points
num_points = 10000

# discretize node by geo
inlet_txyz, outlet_txyz, _, _, top_txyz, bottom_txyz, cylinder_txyz, interior_txyz = \
    sample_data.sample_data(t_num=time_num, t_index=time_index, t_step=t_step, nr_points=num_points)

# interior nodes discre
if use_random_time == True:
    i_t = np.random.uniform(low=time_array[0]/t_star, high=time_array[-1]/t_star, size=len(interior_txyz[:, 0]))
else:
    i_t = interior_txyz[:, 0] / t_star;     print(f"i_t={i_t.shape} {i_t.mean().item():.10f}")
i_x = interior_txyz[:, 1] / xyz_star;   print(f"i_x={i_x.shape} {i_x.mean().item():.10f}")
i_y = interior_txyz[:, 2] / xyz_star;   print(f"i_y={i_y.shape} {i_y.mean().item():.10f}")
i_z = interior_txyz[:, 3] / xyz_star;   print(f"i_z={i_z.shape} {i_z.mean().item():.10f}")

# bc inlet nodes discre
if use_random_time == True:
    b_inlet_t = np.random.uniform(low=time_array[0]/t_star, high=time_array[-1]/t_star, size=len(inlet_txyz[:, 0]))
else:
    b_inlet_t = inlet_txyz[:, 0] / t_star;   print(f"b_inlet_t={b_inlet_t.shape} {b_inlet_t.mean().item():.10f}")
b_inlet_x = inlet_txyz[:, 1] / xyz_star; print(f"b_inlet_x={b_inlet_x.shape} {b_inlet_x.mean().item():.10f}")
b_inlet_y = inlet_txyz[:, 2] / xyz_star; print(f"b_inlet_y={b_inlet_y.shape} {b_inlet_y.mean().item():.10f}")
b_inlet_z = inlet_txyz[:, 3] / xyz_star; print(f"b_inlet_z={b_inlet_z.shape} {b_inlet_z.mean().item():.10f}")

# bc outlet nodes discre
if use_random_time == True:
    b_outlet_t = np.random.uniform(low=time_array[0]/t_star, high=time_array[-1]/t_star, size=len(outlet_txyz[:, 0]))
else:
    b_outlet_t = outlet_txyz[:, 0] / t_star;   print(f"b_outlet_t={b_outlet_t.shape} {b_outlet_t.mean().item():.10f}")
b_outlet_x = outlet_txyz[:, 1] / xyz_star; print(f"b_outlet_x={b_outlet_x.shape} {b_outlet_x.mean().item():.10f}")
b_outlet_y = outlet_txyz[:, 2] / xyz_star; print(f"b_outlet_y={b_outlet_y.shape} {b_outlet_y.mean().item():.10f}")
b_outlet_z = outlet_txyz[:, 3] / xyz_star; print(f"b_outlet_z={b_outlet_z.shape} {b_outlet_z.mean().item():.10f}")

# bc cylinder nodes discre
if use_random_time == True:
    b_cylinder_t = np.random.uniform(low=time_array[0]/t_star, high=time_array[-1]/t_star, size=len(cylinder_txyz[:, 0]))
else:
    b_cylinder_t = cylinder_txyz[:, 0] / t_star;   print(f"b_cylinder_t={b_cylinder_t.shape} {b_cylinder_t.mean().item():.10f}")
b_cylinder_x = cylinder_txyz[:, 1] / xyz_star; print(f"b_cylinder_x={b_cylinder_x.shape} {b_cylinder_x.mean().item():.10f}")
b_cylinder_y = cylinder_txyz[:, 2] / xyz_star; print(f"b_cylinder_y={b_cylinder_y.shape} {b_cylinder_y.mean().item():.10f}")
b_cylinder_z = cylinder_txyz[:, 3] / xyz_star; print(f"b_cylinder_z={b_cylinder_z.shape} {b_cylinder_z.mean().item():.10f}")

# bc front nodes discre
# b_front_t = front_txyz[:, 0] / t_star;   print(f"b_front_t={b_front_t.shape} {b_front_t.mean().item():.10f}")
# b_front_x = front_txyz[:, 1] / xyz_star; print(f"b_front_x={b_front_x.shape} {b_front_x.mean().item():.10f}")
# b_front_y = front_txyz[:, 2] / xyz_star; print(f"b_front_y={b_front_y.shape} {b_front_y.mean().item():.10f}")
# b_front_z = front_txyz[:, 3] / xyz_star; print(f"b_front_z={b_front_z.shape} {b_front_z.mean().item():.10f}")

# bc back nodes discre
# b_back_t = back_txyz[:, 0] / t_star;   print(f"b_back_t={b_back_t.shape} {b_back_t.mean().item():.10f}")
# b_back_x = back_txyz[:, 1] / xyz_star; print(f"b_back_x={b_back_x.shape} {b_back_x.mean().item():.10f}")
# b_back_y = back_txyz[:, 2] / xyz_star; print(f"b_back_y={b_back_y.shape} {b_back_y.mean().item():.10f}")
# b_back_z = back_txyz[:, 3] / xyz_star; print(f"b_back_z={b_back_z.shape} {b_back_z.mean().item():.10f}")

# bc-top nodes discre
if use_random_time == True:
    b_top_t = np.random.uniform(low=time_array[0]/t_star, high=time_array[-1]/t_star, size=len(top_txyz[:, 0]))
else:
    b_top_t = top_txyz[:, 0] / t_star;   print(f"b_top_t={b_top_t.shape} {b_top_t.mean().item():.10f}") # value = [1, 2, 3, 4, 5]
b_top_x = top_txyz[:, 1] / xyz_star; print(f"b_top_x={b_top_x.shape} {b_top_x.mean().item():.10f}")
b_top_y = top_txyz[:, 2] / xyz_star; print(f"b_top_y={b_top_y.shape} {b_top_y.mean().item():.10f}")
b_top_z = top_txyz[:, 3] / xyz_star; print(f"b_top_z={b_top_z.shape} {b_top_z.mean().item():.10f}")

# bc-bottom nodes discre
if use_random_time == True:
    b_bottom_t = np.random.uniform(low=time_array[0]/t_star, high=time_array[-1]/t_star, size=len(bottom_txyz[:, 0]))
else:
    b_bottom_t = bottom_txyz[:, 0] / t_star;   print(f"b_bottom_t={b_bottom_t.shape} {b_bottom_t.mean().item():.10f}") # value = [1, 2, 3, 4, 5]
b_bottom_x = bottom_txyz[:, 1] / xyz_star; print(f"b_bottom_x={b_bottom_x.shape} {b_bottom_x.mean().item():.10f}")
b_bottom_y = bottom_txyz[:, 2] / xyz_star; print(f"b_bottom_y={b_bottom_y.shape} {b_bottom_y.mean().item():.10f}")
b_bottom_z = bottom_txyz[:, 3] / xyz_star; print(f"b_bottom_z={b_bottom_z.shape} {b_bottom_z.mean().item():.10f}")

# bc & interior nodes for nn
inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)
inputbc1 = np.stack((b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z), axis=1)
inputbc2 = np.stack((b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z), axis=1)
inputbc3 = np.stack((b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z), axis=1)
# inputbc4_front = np.stack((b_front_t, b_front_x, b_front_y, b_front_z), axis=1)
# inputbc5_back = np.stack((b_back_t, b_back_x, b_back_y, b_back_z), axis=1)
inputbc6_top = np.stack((b_top_t, b_top_x, b_top_y, b_top_z), axis=1)
inputbc7_bottom = np.stack((b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z), axis=1)

inputic = np.stack((init_t, init_x, init_y, init_z), axis=1)
inputsup = np.stack((sup_t, sup_x, sup_y, sup_z), axis=1)
refsup = np.stack((sup_u, sup_v, sup_w), axis=1)

# N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
pde = psci.pde.NavierStokes(nu=nu, rho=1.0, dim=3, time_dependent=True)

# set bounday condition
bc_inlet_u = psci.bc.Dirichlet("u", rhs=0.1/uvw_star)
bc_inlet_v = psci.bc.Dirichlet("v", rhs=0.0)
bc_inlet_w = psci.bc.Dirichlet("w", rhs=0.0)
bc_cylinder_u = psci.bc.Dirichlet("u", rhs=0.0)
bc_cylinder_v = psci.bc.Dirichlet("v", rhs=0.0)
bc_cylinder_w = psci.bc.Dirichlet("w", rhs=0.0)
bc_outlet_p = psci.bc.Dirichlet("p", rhs=0.0)
# bc_front_u = psci.bc.Dirichlet("u", rhs=0.0)
# bc_front_v = psci.bc.Dirichlet("v", rhs=0.0)
# bc_front_w = psci.bc.Dirichlet("w", rhs=0.0)
# bc_back_u = psci.bc.Dirichlet("u", rhs=0.0)
# bc_back_v = psci.bc.Dirichlet("v", rhs=0.0)
# bc_back_w = psci.bc.Dirichlet("w", rhs=0.0)
bc_top_u = psci.bc.Dirichlet("u", rhs=0.1/uvw_star)
bc_top_v = psci.bc.Dirichlet("v", rhs=0.0)
bc_top_w = psci.bc.Dirichlet("w", rhs=0.0)
bc_bottom_u = psci.bc.Dirichlet("u", rhs=0.1/uvw_star)
bc_bottom_v = psci.bc.Dirichlet("v", rhs=0.0)
bc_bottom_w = psci.bc.Dirichlet("w", rhs=0.0)

# add bounday and boundary condition
pde.set_bc("inlet", bc_inlet_u, bc_inlet_v, bc_inlet_w)
pde.set_bc("cylinder", bc_cylinder_u, bc_cylinder_v, bc_cylinder_w)
pde.set_bc("outlet", bc_outlet_p)
# pde.set_bc("front", bc_front_u, bc_front_v, bc_front_w)
# pde.set_bc("back", bc_back_u, bc_back_v, bc_back_w)
pde.set_bc("top", bc_top_u, bc_top_v, bc_top_w)
pde.set_bc("bottom", bc_bottom_u, bc_bottom_v, bc_bottom_w)

# add initial condition
ic_u = psci.ic.IC("u", rhs=init_u)
ic_v = psci.ic.IC("v", rhs=init_v)
ic_w = psci.ic.IC("w", rhs=init_w)
ic_p = psci.ic.IC("p", rhs=init_p)
pde.set_ic(ic_u, ic_v, ic_w) # 添加ic_p会使ic_loss非常大

# Network
net = psci.network.FCNet(
    num_ins=4, num_outs=4, num_layers=6, hidden_size=hidden_size, activation="tanh")

outeq = net(inputeq)
outbc1 = net(inputbc1)
outbc2 = net(inputbc2)
outbc3 = net(inputbc3)
# outbc4 = net(inputbc4_front)
# outbc5 = net(inputbc5_back)
outbc6 = net(inputbc6_top)
outbc7 = net(inputbc7_bottom)
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
# lossbc4 = psci.loss.BcLoss("front", netout=outbc4)
# lossbc5 = psci.loss.BcLoss("back", netout=outbc5)
lossbc6 = psci.loss.BcLoss("top", netout=outbc6)
lossbc7 = psci.loss.BcLoss("bottom", netout=outbc7)

# ic loss
lossic = psci.loss.IcLoss(netout=outic)

# supervise loss
losssup = psci.loss.DataLoss(netout=outsup[0:3], ref=refsup)

# total loss
loss = losseq1 * eq_wgt + losseq2 * eq_wgt + losseq3 * eq_wgt + losseq4 * eq_wgt + \
    lossbc1 * inlet_wgt + \
    lossbc2 * outlet_wgt + \
    lossbc3 * cylinder_wgt + \
    lossbc6 * top_wgt + \
    lossbc7 * bottom_wgt + \
    lossic * ic_wgt \
    + losssup * sup_wgt

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)
from paddle.regularizer import L2Decay
_lr = psci.optimizer.lr.Cosine(num_epoch, 1, learning_rate = learning_rate, warmup_epoch=w_epoch, by_epoch=True)()
opt = psci.optimizer.Adam(learning_rate=_lr, parameters=net.parameters())


def txyz_nomalization(t_star, xyz_star, txyz_input):
    i_t = txyz_input[:, 0] / t_star;     
    i_x = txyz_input[:, 1] / xyz_star;  
    i_y = txyz_input[:, 2] / xyz_star;   
    i_z = txyz_input[:, 3] / xyz_star;   
    return i_t, i_x, i_y, i_z

lbm_01 = load_vtk([0],  t_step=50, load_uvwp=True, load_txyz=True, name_wt_time=ic_name)[0]
lbm_99 = load_vtk([99], t_step=50, load_uvwp=True, load_txyz=True, name_wt_time=ic_name)[0]
tmp = txyz_nomalization(t_star, xyz_star, lbm_01)
lbm_01[:, 0:4] = np.stack(tmp, axis=1)
tmp = txyz_nomalization(t_star, xyz_star, lbm_99)
lbm_99[:, 0:4] = np.stack(tmp, axis=1)

lbm = [lbm_01, lbm_99]
# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt, lbm=lbm)

# Solve
solution = solver.solve(num_epoch=num_epoch, bs=bs)
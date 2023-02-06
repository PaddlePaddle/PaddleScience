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
from paddlescience.optimizer.lr import Cosine
# cfg = psci.utils.parse_args()

# if cfg is not None:
#     # Geometry
#     npoints = cfg['Geometry']['npoints']
#     seed_num = cfg['Geometry']['seed']
#     sampler_method = cfg['Geometry']['sampler_method']
#     # Time
#     start_time = cfg['Time']['start_time']
#     end_time = cfg['Time']['end_time']
#     time_step = cfg['Time']['time_step']
#     # Network
#     epochs = cfg['Global']['epochs']
#     num_layers = cfg['Model']['num_layers']
#     hidden_size = cfg['Model']['hidden_size']
#     activation = cfg['Model']['activation']
#     # Optimizer
#     learning_rate = cfg['Optimizer']['lr']['learning_rate']
#     # Post-processing
#     solution_filename = cfg['Post-processing']['solution_filename']
#     vtk_filename = cfg['Post-processing']['vtk_filename']
#     checkpoint_path = cfg['Post-processing']['checkpoint_path']
# else:
    # Config
    # paddle.enable_static()
# Geometry
npoints = 10201
seed_num = 42
sampler_method = 'uniform'
# Time
start_time = 0.0 # train: 0.1; predict: 0.0
end_time = 1.5
time_step = 0.1
time_num = int((end_time - start_time + 0.5 * time_step) / time_step) + 1
time_tmp = np.linspace(start_time, end_time, time_num, endpoint=True)
time_array = time_tmp
print(f"time_num = {time_num}, time_array = {time_array}")

# Network
epochs = 20000
num_layers = 10
hidden_size = 50
activation = 'tanh'
# Optimizer
learning_rate = 0.001
learning_rate = Cosine(epochs, 1, learning_rate, warmup_epoch=int(epochs * 0.05), by_epoch=True)()
# Post-processing
solution_filename = 'output_ldc2d_unsteady_train_newapi'
vtk_filename = 'output_ldc2d_unsteady_train_newapi'
checkpoint_path = 'checkpoints'

paddle.seed(seed_num)
np.random.seed(seed_num)

def replicate_t(t_array, data):
    """
    replicate_t
    """
    full_data = None
    t_len = data.shape[0]
    for time in t_array:
        t_extended = np.array([time] * t_len, dtype="float32").reshape((-1, 1)) # [N, 1]
        t_data = np.concatenate((t_extended, data), axis=1) # [N, xyz]->[N, txyz]
        if full_data is None:
            full_data = t_data
        else:
            full_data = np.concatenate((full_data, t_data)) # [N*t_step,txyz]

    return full_data

# set geometry and boundary
geo = psci.geometry.Rectangular(origin=(-0.5, -0.5), extent=(0.5, 0.5))

geo.add_boundary(name="top", criteria=lambda x, y: abs(y - 0.5) < 1e-5)
geo.add_boundary(name="down", criteria=lambda x, y: abs(y + 0.5) < 1e-5)
geo.add_boundary(name="left", criteria=lambda x, y: abs(x + 0.5) < 1e-5)
geo.add_boundary(name="right", criteria=lambda x, y: abs(x - 0.5) < 1e-5)

# discretize geometry
geo_disc = geo.discretize(npoints=npoints, method=sampler_method)

# N-S: Re = 100, V = 1.0, d = 1, nu = rho V d / Re = 1.0 * 1.0 * 1.0 / 100 = 0.01
#      Re = 400, V = 1.0, d = 1, nu = rho V d / Re = 1.0 * 1.0 * 1.0 / 400 = 0.0025
pde = psci.pde.NavierStokes(
    nu=0.01, rho=1.0, dim=2, time_dependent=True, weight=0.0001)
pde.set_time_interval([start_time, end_time])

# define boundary condition
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0)
# weight_top_u = lambda x, y: 1.0 - 20.0 * abs(x)
# bc_top_u = psci.bc.Dirichlet('u', rhs=1.0, weight=weight_top_u)
bc_top_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_down_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_down_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_left_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0.0)
bc_right_u = psci.bc.Dirichlet('u', rhs=0.0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0.0)

# set boundary condition
pde.set_bc("top", bc_top_u, bc_top_v)
pde.set_bc("down", bc_down_u, bc_down_v)
pde.set_bc("left", bc_left_u, bc_left_v)
pde.set_bc("right", bc_right_u, bc_right_v)

# define initial condition
ic_u = psci.ic.IC('u', rhs=0.0)
ic_v = psci.ic.IC('v', rhs=0.0)

# set initial condition
pde.set_ic(ic_u, ic_v)

# discretize pde
# pde_disc = pde.discretize(time_step=time_step, geo_disc=geo_disc)

# Network
# TODO: remove num_ins and num_outs
net = psci.network.FCNet(
    num_ins=3,
    num_outs=3,
    num_layers=num_layers,
    hidden_size=hidden_size,
    activation=activation)
net.initialize("./checkpoint/dynamic_net_params_20000.pdparams")

# eq loss
cords_interior = geo_disc.interior; num_cords = cords_interior.shape[0]; print("num_cords = ", num_cords)
inputeq = replicate_t(time_array, cords_interior)

outeq = net(inputeq)
losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)

# ic loss
inputic = replicate_t([0], cords_interior); print("inputic.shape: ", inputic.shape)
outic = net(inputic)
lossic = psci.loss.IcLoss(netout=outic[:, :2])

# bc loss
inputbc_top = geo_disc.boundary["top"]
inputbc_top = replicate_t(time_array, inputbc_top)
outbc_top = net(inputbc_top)
lossbc_top = psci.loss.BcLoss("top", netout=outbc_top)

inputbc_down = geo_disc.boundary["down"]
inputbc_down = replicate_t(time_array, inputbc_down)
outbc_down = net(inputbc_down)
lossbc_down = psci.loss.BcLoss("down", netout=outbc_down)

inputbc_left = geo_disc.boundary["left"]
inputbc_left = replicate_t(time_array, inputbc_left)
outbc_left = net(inputbc_left)
lossbc_left = psci.loss.BcLoss("left", netout=outbc_left)

inputbc_right = geo_disc.boundary["right"]
inputbc_right = replicate_t(time_array, inputbc_right)
outbc_right = net(inputbc_right)
lossbc_right = psci.loss.BcLoss("right", netout=outbc_right)

# total loss
loss = losseq1 + \
    losseq2 + \
    losseq3 + \
    10.0 * lossbc_top + \
    10.0 * lossbc_down + \
    10.0 * lossbc_left + \
    10.0 * lossbc_right + \
    10.0 * lossic

# Loss
# loss = psci.loss.L2(p=2)

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(
    learning_rate=learning_rate, parameters=net.parameters())
# inputeq[:num_cords, 0] = 0.0

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)
# solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
# solution = solver.solve(num_epoch=epochs)
solution = solver.predict()
for i in range(len(solution)):
    print(f"solution[{i}]={solution[i].shape}")

# Save result to vtk
for i in range(time_num):
    psci.visu.__save_vtk_raw(filename=f"./vtk/ldc2d_output_u_time{i}", cordinate=geo_disc.interior, data=solution[0][i*num_cords:(i+1)*num_cords])
# psci.visu.save_vtk(
#     filename=vtk_filename,
#     time_array=pde_disc.time_array,
#     geo_disc=pde_disc.geometry,
#     data=solution)

# psci.visu.save_npy(
#     filename=solution_filename,
#     time_array=pde_disc.time_array,
#     geo_disc=pde_disc.geometry,
#     data=solution)

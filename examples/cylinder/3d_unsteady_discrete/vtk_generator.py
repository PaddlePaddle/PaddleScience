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

paddle.seed(42)
np.random.seed(42)

def load_input(t_star, xyz_star, name_wt_time):
    txyz_uvwpe_input = load_vtk([0], 0, load_uvwp=True, load_txyz=True, name_wt_time=name_wt_time)[0]

    num_time = time_tmp.shape[0]
    num_nodes = txyz_uvwpe_input.shape[0] # nodes number of every time
    num_nodes_all_time = num_time * num_nodes
    it = np.zeros(num_nodes_all_time)
    ix =  np.zeros(num_nodes_all_time)
    iy =  np.zeros(num_nodes_all_time)
    iz =  np.zeros(num_nodes_all_time)

    for i, time in enumerate(time_tmp):
        it[i * num_nodes : (i + 1) * num_nodes] = time
        ix[i * num_nodes : (i + 1) * num_nodes] = txyz_uvwpe_input[:, 1]
        iy[i * num_nodes : (i + 1) * num_nodes] = txyz_uvwpe_input[:, 2]
        iz[i * num_nodes : (i + 1) * num_nodes] = txyz_uvwpe_input[:, 3]

    i_t = it / t_star;     print(f"i_t={i_t.shape} {i_t.mean().item():.10f}")
    i_x = ix / xyz_star;   print(f"i_x={i_x.shape} {i_x.mean().item():.10f}")
    i_y = iy / xyz_star;   print(f"i_y={i_y.shape} {i_y.mean().item():.10f}")
    i_z = iz / xyz_star;   print(f"i_z={i_z.shape} {i_z.mean().item():.10f}")
    return i_t, i_x, i_y, i_z, num_nodes


def xyz_denomalization(i_x, i_y, i_z, xyz_star, num_time):
    # only coord at start time is needed
    n = int(i_x.shape[0] / num_time)
    i_x = i_x.astype("float32")
    i_y = i_y.astype("float32")
    i_z = i_z.astype("float32")

    # denormalize back
    i_x = i_x * xyz_star
    i_y = i_y * xyz_star
    i_z = i_z * xyz_star
    cord = np.stack((i_x[0:n], i_y[0:n], i_z[0:n]), axis=1)
    return cord


def uvwp_denomalization(solution, p_star, uvw_star):
    # denormalization
    for i in range(len(solution)):
        solution[i][:, 0:1] = solution[i][:, 0:1] * uvw_star
        solution[i][:, 1:2] = solution[i][:, 1:2] * uvw_star
        solution[i][:, 2:3] = solution[i][:, 2:3] * uvw_star
        solution[i][:, 3:4] = solution[i][:, 3:4] * p_star


if __name__=="__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))
    net_ref = dirname + r'/checkpoint/ref_dynamic_net_params_100000.pdparams'
    net_mini_batch = dirname + r'/checkpoint/dynamic_net_params_40000.pdparams'
    net_width_ref = 50
    net_width_mini_batch = 512
    net_width = net_width_ref

    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    config = psci.utils.get_config(fname=dirname + r'/config.yaml', config_index="hyper parameters")
    name_wt_time = dirname + r'/data/LBM_result_20_steps/cylinder3d_2023_1_31_LBM_'

    Re = 3900
    U0 = 0.1
    Dcylinder = 80.0
    rho = 1.0
    nu = rho * U0 * Dcylinder / Re

    t_star = Dcylinder / U0 # 800
    xyz_star = Dcylinder    # 80
    uvw_star = U0           # 0.1ß
    p_star = rho * U0 * U0  # 0.01

    # time arrayß
    ic_t = 200000
    t_start = 200050
    t_end = 201000
    t_step = 50
    time_num = int((t_end - t_start) / t_step) + 1
    time_list = np.linspace(int((t_start - ic_t) / t_step), int((t_end - ic_t) / t_step), time_num, endpoint=True).astype(int)
    time_tmp = time_list * t_step
    time_index = np.random.choice(time_list, int(time_num / 2.5), replace=False)
    time_index.sort()
    time_array = time_index * t_step
    print(f"time_num = {time_num}")
    print(f"time_list = {time_list}")
    print(f"time_tmp = {time_tmp}")
    print(f"time_index = {time_index}")
    print(f"time_array = {time_array}")
    num_time = time_tmp.shape[0]

    lbm_uvwp = load_vtk(time_list=time_list, t_step=t_step, load_uvwp=True, name_wt_time=name_wt_time)

    # load baseline nodes coordinates
    i_t, i_x, i_y, i_z, n = load_input(t_star, xyz_star, name_wt_time=name_wt_time)

    # eq cord
    inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = psci.pde.NavierStokes(nu=0.00205, rho=1.0, dim=3, time_dependent=True)

    # Network
    net = psci.network.FCNet(
        num_ins=4, num_outs=4, num_layers=6, hidden_size=net_width, activation="tanh")
    outeq = net(inputeq)

    # Initialize Net
    net.initialize(net_ref)

    # eq loss(decoupling refactorization is on the way)
    losseq1 = psci.loss.EqLoss(pde.equations[0], netout=outeq)
    losseq2 = psci.loss.EqLoss(pde.equations[1], netout=outeq)
    losseq3 = psci.loss.EqLoss(pde.equations[2], netout=outeq)
    losseq4 = psci.loss.EqLoss(pde.equations[3], netout=outeq)

    # Algorithms
    fake_loss = losseq1 + losseq2 + losseq3 + losseq4
    algo = psci.algorithm.PINNs(net=net, loss=fake_loss)

    # Solver
    solver = psci.solver.Solver(pde=pde, algo=algo)

    # Solve
    solution = solver.predict()

    # denormalization
    cord = xyz_denomalization(i_x, i_y, i_z, xyz_star, num_time)
    uvwp_denomalization(solution, p_star, uvw_star) # modify [solution]

    # LBM baseline, output Error 
    print("/*------------------ Quantitative analysis : LBM baseline error ------------------*/")
    residual = []
    for i in range(num_time): 
        temp_list = lbm_uvwp[i][:, 4:8] - solution[0][i*n:(i+1)*n]
        residual.append(np.absolute(np.array(temp_list)))
        print(
        f"{time_list[i]} \
            time = {time_tmp[i]} s, \
            sum = {(np.array(residual[i])).sum(axis=0)}，\
            mean = {(np.array(residual[i])).mean(axis=0)}, \
            median = {np.median(np.array(residual[i]), axis=0)}")
        psci.visu.__save_vtk_raw(filename = dirname + f"/vtk/0302_error_{i+1}", cordinate=cord, data=temp_list)


    # Output VTK
    print("/*------------------     Output VTK : Result visualization     ------------------*/")
    for i in range(num_time):
        print(cord.shape, solution[0][i*n:(i+1)*n].shape)
        psci.visu.__save_vtk_raw(filename = dirname + f"/vtk/0302_predict_{i+1}", cordinate=cord, data=solution[0][i*n:(i+1)*n])
        

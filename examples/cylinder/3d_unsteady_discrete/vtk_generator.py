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
import csv
import paddle
import numpy as np
import paddlescience as psci
from load_lbm_data import load_vtk, load_msh, write_vtu

paddle.seed(42)
np.random.seed(42)


def load_input(t_star, xyz_star, file_name, time_tmp, not_mesh):
    if not_mesh == True:
        txyz_uvwpe_input = load_vtk(
            [0], 0, load_uvwp=True, load_txyz=True, name_wt_time=file_name)[0]
        input_mesh = 0
    else:
        txyz_uvwpe_input, input_mesh = load_msh(file_name)
    num_time = time_tmp.shape[0]
    num_nodes = txyz_uvwpe_input.shape[0]  # nodes number of every time
    num_nodes_all_time = num_time * num_nodes
    it = np.zeros(num_nodes_all_time)
    ix = np.zeros(num_nodes_all_time)
    iy = np.zeros(num_nodes_all_time)
    iz = np.zeros(num_nodes_all_time)

    for i, time in enumerate(time_tmp):
        it[i * num_nodes:(i + 1) * num_nodes] = time
        ix[i * num_nodes:(i + 1) * num_nodes] = txyz_uvwpe_input[:, 1]
        iy[i * num_nodes:(i + 1) * num_nodes] = txyz_uvwpe_input[:, 2]
        iz[i * num_nodes:(i + 1) * num_nodes] = txyz_uvwpe_input[:, 3]

    i_t = it / t_star
    print(f"i_t={i_t.shape} {i_t.mean().item():.10f}")
    i_x = ix / xyz_star
    print(f"i_x={i_x.shape} {i_x.mean().item():.10f}")
    i_y = iy / xyz_star
    print(f"i_y={i_y.shape} {i_y.mean().item():.10f}")
    i_z = iz / xyz_star
    print(f"i_z={i_z.shape} {i_z.mean().item():.10f}")
    return i_t, i_x, i_y, i_z, num_nodes, input_mesh


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


def write_error(baseline, solution, err_min, err_index):
    # update err_min & err_index
    new_err_min = err_min
    new_err_index = err_index
    err = baseline[0][:, 4:8] - solution[0][0:n]
    err_sum = (np.absolute(err)).sum(axis=0)
    err_mean = (np.absolute(err)).mean(axis=0)
    if err_min > err_sum[0]:
        new_err_min = err_sum[0]
        new_err_index = i
    print(f'epoch = {int(i)}', f'  sum = {err_sum}')

    # write csv
    with open(dirname + r'/output/err.csv', 'a') as file:
        w = csv.writer(file)
        if err_index == -1:
            w.writerow(['epoch', 'error u', 'error v', 'error w', 'error p'])
        w.writerow([int(i)] + err_mean.tolist())

    return new_err_min, new_err_index


def net_predict(net_ini, pde, net_width, inputeq):
    # Network
    net = psci.network.FCNet(
        num_ins=4,
        num_outs=4,
        num_layers=6,
        hidden_size=net_width,
        activation="tanh")
    outeq = net(inputeq)

    # Initialize Net
    net.initialize(net_ini)

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
    return solution


if __name__ == "__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))

    # Net Setting
    net_1 = dirname + r'/checkpoint/0217_origional.pdparams'
    net_2 = dirname + r'/checkpoint/0302_mini_batch.pdparams'
    net_3 = dirname + r'/checkpoint/0307_4e5.pdparams'
    net_4 = dirname + r'/checkpoint/0315_99steps_335.pdparams'

    net_ini = net_4
    net_width = 512

    not_mesh = True  # if u do prediction by mesh, pick False

    ref_file = dirname + r'/data/LBM_result/cylinder3d_2023_1_31_LBM_'
    msh_file = dirname + r'/data/3d_cylinder.msh'

    Re = 3900
    U0 = 0.1
    Dcylinder = 80.0
    rho = 1.0
    nu = rho * U0 * Dcylinder / Re

    t_star = Dcylinder / U0  # 800
    xyz_star = Dcylinder  # 80
    uvw_star = U0  # 0.1
    p_star = rho * U0 * U0  # 0.01

    # time array
    ic_t = 200000
    t_start = 200050
    t_end = 204950
    t_step = 50
    time_num = int((t_end - t_start) / t_step) + 1
    time_list = np.linspace(
        int((t_start - ic_t) / t_step),
        int((t_end - ic_t) / t_step),
        time_num,
        endpoint=True).astype(int)
    time_tmp = time_list * t_step
    num_time = time_tmp.shape[0]

    # load coordinates of prediction nodes
    if not_mesh == True:
        lbm_uvwp = load_vtk(
            time_list=time_list,
            t_step=t_step,
            load_uvwp=True,
            name_wt_time=ref_file)
        input_file = ref_file
    else:
        input_file = msh_file

    i_t, i_x, i_y, i_z, n, input_mesh = load_input(
        t_star,
        xyz_star,
        file_name=input_file,
        time_tmp=time_tmp,
        not_mesh=not_mesh)
    cord = xyz_denomalization(i_x, i_y, i_z, xyz_star, num_time)

    # eq cord
    inputeq = np.stack((i_t, i_x, i_y, i_z), axis=1)

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = psci.pde.NavierStokes(nu=nu, rho=1.0, dim=3, time_dependent=True)

    if not_mesh == True:
        #Find min error
        print(
            "/*------------------     Saved Models : find min err       ------------------*/"
        )
        time_checklist = np.arange(1000, 401000, 1000)  #1000 minimum
        err_min = float("inf")
        err_index = -1
        lbm_99 = load_vtk(
            time_list=[99],
            t_step=t_step,
            load_uvwp=True,
            name_wt_time=ref_file)
        e_input = load_input(
            t_star,
            xyz_star,
            file_name=ref_file,
            time_tmp=np.array([4950]),
            not_mesh=True)
        for i in time_checklist:
            net_ini = dirname + r'/99steps_checkpoint/dynamic_net_params_' + str(
                int(i)) + '.pdparams'
            solution = net_predict(
                net_ini, pde, net_width, np.stack(
                    e_input[0:4], axis=1))
            uvwp_denomalization(solution, p_star,
                                uvw_star)  # modify [solution]
            err_min, err_index = write_error(
                baseline=lbm_99,
                solution=solution,
                err_min=err_min,
                err_index=err_index)
        net_ini = dirname + r'/99steps_checkpoint/dynamic_net_params_' + str(
            int(err_index)) + '.pdparams'
        print(f'*best epoch = {int(err_index)}', f'  sum = {err_min}')

    # evaluate with the best model
    solution = net_predict(net_ini, pde, net_width, inputeq)
    uvwp_denomalization(solution, p_star, uvw_star)

    print(
        "/*------------------ Quantitative analysis : LBM baseline error -----------------*/"
    )
    if not_mesh == True:
        # LBM baseline, output Error 
        residual = []
        for i in range(num_time):
            temp_list = lbm_uvwp[i][:, 4:8] - solution[0][i * n:(i + 1) * n]
            residual.append(np.absolute(np.array(temp_list)))
            print(f"{time_list[i]} \
                time = {time_tmp[i]} s, \
                sum = {(np.array(residual[i])).sum(axis=0)}，\
                mean = {(np.array(residual[i])).mean(axis=0)}, \
                median = {np.median(np.array(residual[i]), axis=0)}")
            # psci.visu.__save_vtk_raw(filename = dirname + f"/vtk/0302_error_{i+1}", cordinate=cord, data=temp_list)  # output error being displayed in paraview

        # Output VTK
    print(
        "/*------------------     Output VTK : Result visualization     ------------------*/"
    )
    for i in range(num_time):
        if not_mesh == False:
            write_vtu(
                file=dirname + f"/vtk_mesh/0302_predict_{i+1}.vtu",
                mesh=input_mesh,
                solution=solution[0][i * n:(i + 1) * n])
        else:
            psci.visu.__save_vtk_raw(
                filename=dirname + f"/vtk/0302_predict_{i+1}",
                cordinate=cord,
                data=solution[0][i * n:(i + 1) * n])

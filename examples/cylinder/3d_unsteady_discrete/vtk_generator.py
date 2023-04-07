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

import csv
import os
import sys
from pickle import TRUE

import numpy as np
import paddle
from load_baseline_data import Input
from load_baseline_data import Label
from load_baseline_data import load_msh
from load_baseline_data import load_vtk
from load_baseline_data import write_vtu

import paddlescience as psci
import ppsci as new_psci

paddle.seed(42)
np.random.seed(42)


def load_input(t_star, xyz_star, file_name, time_tmp, not_mesh):
    """load inputs, combine by time steps, decide load from mesh or not

    Args:
        t_star (_type_): _description_
        xyz_star (_type_): _description_
        file_name (_type_): _description_
        time_tmp (_type_): _description_
        not_mesh (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not_mesh == True:
        txyz_uvwpe_input = load_vtk(
            [0], 0, load_uvwp=True, load_txyz=True, name_wt_time=file_name
        )[0]
        input_mesh = 0
    else:
        txyz_uvwpe_input, input_mesh = load_msh(file_name)
    num_time = time_tmp.shape[0]
    num_nodes = txyz_uvwpe_input.shape[0]  # nodes number of every time step
    num_nodes_all_time = num_time * num_nodes
    it = np.zeros((num_nodes_all_time, 1)).astype(np.float32)
    ix = np.zeros((num_nodes_all_time, 1)).astype(np.float32)
    iy = np.zeros((num_nodes_all_time, 1)).astype(np.float32)
    iz = np.zeros((num_nodes_all_time, 1)).astype(np.float32)

    for i, time in enumerate(time_tmp):
        it[i * num_nodes : (i + 1) * num_nodes] = time
        ix[i * num_nodes : (i + 1) * num_nodes] = txyz_uvwpe_input[:, 1]
        iy[i * num_nodes : (i + 1) * num_nodes] = txyz_uvwpe_input[:, 2]
        iz[i * num_nodes : (i + 1) * num_nodes] = txyz_uvwpe_input[:, 3]

    i_t = it / t_star
    print(f"i_t={i_t.shape} {i_t.mean().item():.10f}")
    i_x = ix / xyz_star
    print(f"i_x={i_x.shape} {i_x.mean().item():.10f}")
    i_y = iy / xyz_star
    print(f"i_y={i_y.shape} {i_y.mean().item():.10f}")
    i_z = iz / xyz_star
    print(f"i_z={i_z.shape} {i_z.mean().item():.10f}")
    return i_t, i_x, i_y, i_z, num_nodes, input_mesh


def xyz_denormalization(i_x, i_y, i_z, xyz_star, num_time):
    """spatial denomalization

    Args:
        i_x (_type_): _description_
        i_y (_type_): _description_
        i_z (_type_): _description_
        xyz_star (_type_): _description_
        num_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    # only coord at start time is needed
    n = int(i_x.shape[0] / num_time)
    i_x = i_x.astype("float32")
    i_y = i_y.astype("float32")
    i_z = i_z.astype("float32")

    # denormalize back
    i_x = i_x * xyz_star
    i_y = i_y * xyz_star
    i_z = i_z * xyz_star
    cord = {Input.x: list(i_x), Input.y: list(i_y), Input.z: list(i_z)}
    return cord


def uvwp_denormalization(solution, p_star, uvw_star):
    """result denormalization

    Args:
        solution (_type_): _description_
        p_star (_type_): _description_
        uvw_star (_type_): _description_
    """
    # denormalization
    solution["u"] = solution["u"] * uvw_star
    solution["v"] = solution["v"] * uvw_star
    solution["w"] = solution["w"] * uvw_star
    solution["p"] = solution["p"] * p_star


def write_error(baseline, solution, err_min, err_index):
    """write error for train & validation loss

    Args:
        baseline (_type_): _description_
        solution (_type_): _description_
        err_min (_type_): _description_
        err_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    # update err_min & err_index
    new_err_min = err_min
    new_err_index = err_index
    err = baseline[0][:, 4:8] - solution[0][0:n]
    err_sum = (np.absolute(err)).sum(axis=0)
    err_mean = (np.absolute(err)).mean(axis=0)
    if err_min > err_sum[0]:
        new_err_min = err_sum[0]
        new_err_index = i
    print(f"epoch = {int(i)}", f"  sum = {err_sum}")

    # write csv
    with open(dirname + r"/output/err.csv", "a") as file:
        w = csv.writer(file)
        if err_index == -1:
            w.writerow(["epoch", "error u", "error v", "error w", "error p"])
        w.writerow([int(i)] + err_mean.tolist())

    return new_err_min, new_err_index


def net_predict(net_ini, pde, input, config):
    """do net initialization and predict solution

    Args:
        net_ini (_type_): _description_
        pde (_type_): _description_
        net_width (_type_): _description_
        inputeq (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Network

    NUMBER_OF_SPLITS = 20
    model = new_psci.arch.MLP(
        input_keys=["t", "x", "y", "z"],
        output_keys=["u", "v", "w", "p"],
        num_layers=config["layer number"],
        hidden_size=config["hidden size"],
        activation=config["activation function"],
        skip_connection=False,
        weight_norm=False,
    )

    train_solver = new_psci.solver.Solver(
        mode="train",
        model=model,
        equation=pde,
        pretrained_model_path=net_ini,
    )
    solution = train_solver.predict(input, splited_shares=NUMBER_OF_SPLITS)

    return solution


if __name__ == "__main__":
    # set working direcoty to this example folder
    dirname = os.path.dirname(os.path.abspath(__file__))

    # Net Setting
    net_1 = dirname + r"/checkpoint/0217_origional.pdparams"
    net_2 = dirname + r"/checkpoint/0302_mini_batch.pdparams"
    net_3 = dirname + r"/checkpoint/0307_4e5.pdparams"
    net_4 = dirname + r"/checkpoint/0315_99steps_335.pdparams"
    net_5 = dirname + r"/checkpoints/epoch_301000"

    net_ini = net_5
    net_width = 512

    not_mesh = True  # if do prediction by mesh, pick False
    find_min_error = False  # if calculate error and find the minimum, pick True

    ref_file = dirname + r"/data/LBM_result/cylinder3d_2023_1_31_LBM_"
    msh_file = dirname + r"/data/3d_cylinder.msh"

    Re = 3900
    U0 = 0.1  # characteristic velocity
    L = 80.0  # characteristic length
    RHO = 1.0
    NU = RHO * U0 * L / Re

    t_star = L / U0  # 800
    xyz_star = L  # 80
    uvw_star = U0  # 0.1
    p_star = RHO * U0 * U0  # 0.01

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
        endpoint=True,
    ).astype(int)
    time_tmp = time_list * t_step
    num_time = time_tmp.shape[0]

    config = psci.utils.get_config(
        fname=dirname + r"/config.yaml", config_index="hyper parameters"
    )

    # load coordinates of prediction nodes
    if not_mesh == True:
        _, lbm_uvwp = load_vtk(
            time_list=time_list,
            t_step=t_step,
            load_uvwp=True,
            name_wt_time=ref_file,
            load_in_dict_shape=True,
        )  # using referece sampling points coordinates[t\x\y\z] as input
        input_file = ref_file
    else:
        input_file = msh_file  # using mesh nodes coordinates[t\x\y\z] as input

    i_t, i_x, i_y, i_z, n, input_mesh = load_input(
        t_star, xyz_star, file_name=input_file, time_tmp=time_tmp, not_mesh=not_mesh
    )
    cord = xyz_denormalization(i_x, i_y, i_z, xyz_star, num_time)

    # sampling points' coodinates for prediction
    input = {"t": i_t, "x": i_x, "y": i_y, "z": i_z}

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = new_psci.equation.NavierStokes(nu=NU, rho=RHO, dim=3, time=True)

    if not_mesh == True and find_min_error == True:
        # Find min error
        print(
            "/*------------------     Saved Models : find min err       ------------------*/"
        )
        time_checklist = np.arange(1000, 401000, 1000)  # 1000 minimum
        err_min = float("inf")
        err_index = -1
        lbm_99 = load_vtk(
            time_list=[99], t_step=t_step, load_uvwp=True, name_wt_time=ref_file
        )
        e_input = load_input(
            t_star,
            xyz_star,
            file_name=ref_file,
            time_tmp=np.array([4950]),
            not_mesh=True,
        )
        for i in time_checklist:
            net_ini = dirname + r"/checkpoints/epoch_" + str(int(i)) + ".pdparams"
            solution = net_predict(net_ini, pde, np.stack(e_input[0:4], axis=1), config)
            uvwp_denormalization(solution, p_star, uvw_star)  # modify [solution]
            err_min, err_index = write_error(
                baseline=lbm_99, solution=solution, err_min=err_min, err_index=err_index
            )
        net_ini = (
            dirname
            + r"/99steps_checkpoint/dynamic_net_params_"
            + str(int(err_index))
            + ".pdparams"
        )
        print(f"*best epoch = {int(err_index)}", f"  sum = {err_min}")

    # evaluate with the best model
    solution = net_predict(net_ini, pde, input, config)
    uvwp_denormalization(solution, p_star, uvw_star)

    print(
        "/*------------------ Quantitative analysis : LBM baseline error -----------------*/"
    )
    if not_mesh == True:
        # LBM baseline, output Error
        err_dict = {key: [] for key in Label}
        for i in range(num_time):
            for key in solution.keys():
                err_dict[Label[key]].append(
                    lbm_uvwp[i][Label[key]]
                    - solution[key][
                        i * n : (i + 1) * n
                    ]  # n : nodes number per time step
                )
            print(
                f"{time_list[i]} \
                time = {time_tmp[i]} s, \
                sum = {(np.absolute(err_dict[Label.u][i])).sum(axis=0)}，\
                mean = {(np.absolute(err_dict[Label.u][i])).mean(axis=0)}, \
                median = {np.median(np.absolute(err_dict[Label.u][i]), axis=0)}"
            )
            # psci.visu.__save_vtk_raw(filename = dirname + f"/vtk/0302_error_{i+1}", cordinate=cord, data=temp_list)  # output error being displayed in paraview

        # Output VTK
    print(
        "/*------------------     Output VTK : Result visualization     ------------------*/"
    )
    for i in range(num_time):
        if not_mesh == False:
            write_vtu(
                file=dirname + f"/vtk_mesh/0302_predict_with_mesh_{i+1}.vtu",
                mesh=input_mesh,
                coordinate=None,
                label={
                    key: solution[key][i * n : (i + 1) * n] for key in solution.keys()
                },  # n : nodes number per time step
            )
        else:
            write_vtu(
                filename=dirname + f"/vtk/0302_predict_{i+1}.vtu",
                mesh=None,
                coordinates={
                    key: cord[key][i * n : (i + 1) * n] for key in cord.keys()
                },
                label={
                    key: solution[key][i * n : (i + 1) * n] for key in solution.keys()
                },  # n : nodes number per time step
            )

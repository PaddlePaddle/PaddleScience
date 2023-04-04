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
"""
Created in Mar. 2023
@author: Guan Wang
"""
import os
import sys

import numpy as np
import paddle
from load_baseline_data import Input
from load_baseline_data import Output
from load_baseline_data import load_msh
from load_baseline_data import load_sample_vtk
from load_baseline_data import load_vtk
from sympy import Q
from sympy import Symbol

import paddlescience as psci
import ppsci as new_psci


def txyz_nomalization(t_factor, xyz_factor, txyz_input):
    """_summary_

    Args:
        t_factor (_type_): _description_
        xyz_factor (_type_): _description_
        txyz_input (_type_): _description_

    Returns:
        _type_: _description_
    """
    time = txyz_input[:, 0] / t_factor
    x_cord = txyz_input[:, 1] / xyz_factor
    y_cord = txyz_input[:, 2] / xyz_factor
    z_cord = txyz_input[:, 3] / xyz_factor
    return time, x_cord, y_cord, z_cord


def normlization(factor, input):
    normolized_input = {}
    if type(input) is list and type(factor) is list:
        for i, x in enumerate(input):
            normolized_input.append(x / factor[i])
    elif type(input) is dict and type(factor) is dict:
        for key, value in input.items():
            normolized_input[key] = value / factor[key]
    return normolized_input


def print_for_check(check_list, varname_list):
    """_summary_

    Args:
        check_list (_type_): _description_
    """
    for _, (term, name_str) in enumerate(zip(check_list, varname_list)):
        print(f"{name_str}={term.shape} {term.mean().item():.10f}")


def normalized_bc(origion_list, t_factor, xyz_factor, random_time):
    """normalize bc data time and coordinates

    Args:
        origion_list (_type_): _description_
        t_factor (_type_): _description_
        xyz_factor (_type_): _description_
        random_time (_type_): _description_

    Returns:
        _type_: _description_
    """
    if random_time is True:
        time = np.random.uniform(
            low=time_array[0] / t_factor,
            high=time_array[-1] / t_factor,
            size=len(origion_list[:, 0]),
        )
    else:
        time = origion_list[:, 0] / t_factor
    x_cord = origion_list[:, 1] / xyz_factor
    y_cord = origion_list[:, 2] / xyz_factor
    z_cord = origion_list[:, 3] / xyz_factor
    return time, x_cord, y_cord, z_cord


if __name__ == "__main__":
    RENOLDS_NUMBER = 3900
    U0 = 0.1
    D_CYLINDER = 80.0
    RHO = 1.0
    NU = RHO * U0 * D_CYLINDER / RENOLDS_NUMBER

    T_STAR = D_CYLINDER / U0  # 800
    XYZ_STAR = D_CYLINDER  # 80
    UVW_STAR = U0  # 0.1
    p_star = RHO * U0 * U0  # 0.01

    # read configuration file : config.yaml
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    config = psci.utils.get_config(
        fname=dirname + r"/config.yaml", config_index="hyper parameters"
    )
    if config is not None:
        # number of epoch
        num_epoch = config["number of epoch"]
        learning_rate = config["learning rate"]
        hidden_size = config["hidden size"]
        num_layers = config["layer number"]
        activation_func = config["activation function"]
        # batch size
        bs = {}
        bs["interior"] = config["batch size"]["interior"]

        bs["inlet"] = config["batch size"]["inlet"]
        bs["outlet"] = config["batch size"]["outlet"]
        bs["cylinder"] = config["batch size"]["cylinder"]
        bs["top"] = config["batch size"]["top"]
        bs["bottom"] = config["batch size"]["bottom"]

        bs["ic"] = config["batch size"]["initial condition"]
        bs["supervised"] = config["batch size"]["supervised"]

        # losses weight
        ic_wgt = config["weight of losses"]["initial condition"]
        eq_wgt = config["weight of losses"]["pde"]

        front_wgt = config["weight of losses"]["front"]
        back_wgt = config["weight of losses"]["back"]
        inlet_wgt = config["weight of losses"]["left inlet"]
        outlet_wgt = config["weight of losses"]["right outlet"]
        top_wgt = config["weight of losses"]["top"]
        bottom_wgt = config["weight of losses"]["bottom"]
        cylinder_wgt = config["weight of losses"]["cylinder"]

        sup_wgt = config["weight of losses"]["supervised"]

        # simulated annealing
        w_epoch = config["Simulated Annealing"]["warm epochs"]

        # epoch number
        seed_number = config["random seed"]
    else:
        print("Error : mssing configure files !")

    USE_RANDOM_TIME = False

    # fix the random seed
    paddle.seed(seed_number)
    np.random.seed(seed_number)

    # time array
    INITIAL_TIME = 200000
    START_TIME = 200050
    END_TIME = 204950
    TIME_STEP = 50
    TIME_NUMBER = int((END_TIME - START_TIME) / TIME_STEP) + 1
    time_list = np.linspace(
        int((START_TIME - INITIAL_TIME) / TIME_STEP),
        int((END_TIME - INITIAL_TIME) / TIME_STEP),
        TIME_NUMBER,
        endpoint=True,
    ).astype(int)
    time_tmp = time_list * TIME_STEP
    time_index = np.random.choice(time_list, int(TIME_NUMBER / 2.5), replace=False)
    time_index.sort()
    time_array = time_index * TIME_STEP
    print(f"TIME_NUMBER = {TIME_NUMBER}")
    print(f"time_list = {time_list}")
    print(f"time_tmp = {time_tmp}")
    print(f"time_index = {time_index}")
    print(f"time_array = {time_array}")

    # initial value
    ic_name = dirname + r"/data/LBM_result/cylinder3d_2023_1_31_LBM_"
    txyz_uvwpe_ic = load_vtk(
        [0], t_step=TIME_STEP, load_uvwp=True, load_txyz=True, name_wt_time=ic_name
    )[0]
    init_t = txyz_uvwpe_ic[:, 0] / T_STAR
    init_x = txyz_uvwpe_ic[:, 1] / XYZ_STAR
    init_y = txyz_uvwpe_ic[:, 2] / XYZ_STAR
    init_z = txyz_uvwpe_ic[:, 3] / XYZ_STAR
    init_u = txyz_uvwpe_ic[:, 4] / UVW_STAR
    init_v = txyz_uvwpe_ic[:, 5] / UVW_STAR
    init_w = txyz_uvwpe_ic[:, 6] / UVW_STAR
    init_p = txyz_uvwpe_ic[:, 7] / p_star
    print_for_check(
        [init_t, init_x, init_y, init_z, init_u, init_v, init_w, init_p],
        [
            "init_t",
            "init_x",
            "init_y",
            "init_z",
            "init_u",
            "init_v",
            "init_w",
            "init_p",
        ],
    )

    # supervised data
    sup_name = dirname + r"/data/sup_data/supervised_"
    txyz_uvwpe_s_new = load_vtk(
        time_index,
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=sup_name,
    )
    txyz_uvwpe_s = np.zeros((0, 8, 1)).astype(np.float32)
    for x in txyz_uvwpe_s_new:
        txyz_uvwpe_s = np.concatenate((txyz_uvwpe_s, x[:, :]), axis=0)
    sup_t = txyz_uvwpe_s[:, 0] / T_STAR
    sup_x = txyz_uvwpe_s[:, 1] / XYZ_STAR
    sup_y = txyz_uvwpe_s[:, 2] / XYZ_STAR
    sup_z = txyz_uvwpe_s[:, 3] / XYZ_STAR
    sup_u = txyz_uvwpe_s[:, 4] / UVW_STAR
    sup_v = txyz_uvwpe_s[:, 5] / UVW_STAR
    sup_w = txyz_uvwpe_s[:, 6] / UVW_STAR
    sup_p = txyz_uvwpe_s[:, 7] / p_star
    print_for_check(
        [sup_t, sup_x, sup_y, sup_z, sup_u, sup_v, sup_w, sup_p],
        ["sup_t", "sup_x", "sup_y", "sup_z", "sup_u", "sup_v", "sup_w", "sup_p"],
    )

    # num of interior points
    NUM_POINTS = 10000
    inlet_txyz = load_sample_vtk(dirname + r"/data/sample_points/inlet_txyz.vtu")
    outlet_txyz = load_sample_vtk(dirname + r"/data/sample_points/outlet_txyz.vtu")
    top_txyz = load_sample_vtk(dirname + r"/data/sample_points/top_txyz.vtu")
    bottom_txyz = load_sample_vtk(dirname + r"/data/sample_points/bottom_txyz.vtu")
    cylinder_txyz = load_sample_vtk(dirname + r"/data/sample_points/cylinder_txyz.vtu")
    interior_txyz = load_sample_vtk(dirname + r"/data/sample_points/interior_txyz.vtu")

    # interior nodes discre
    i_t, i_x, i_y, i_z = normalized_bc(
        interior_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR, random_time=USE_RANDOM_TIME
    )
    print_for_check([i_t, i_x, i_y, i_z], ["i_t", "i_x", "i_y", "i_z"])

    # bc inlet nodes discre
    b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z = normalized_bc(
        inlet_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR, random_time=USE_RANDOM_TIME
    )
    print_for_check(
        [b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z],
        ["b_inlet_t", "b_inlet_x", "b_inlet_y", "b_inlet_z"],
    )

    # bc outlet nodes discre
    b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z = normalized_bc(
        outlet_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR, random_time=USE_RANDOM_TIME
    )
    print_for_check(
        [b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z],
        ["b_outlet_t", "b_outlet_x", "b_outlet_y", "b_outlet_z"],
    )

    # bc cylinder nodes discre
    b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z = normalized_bc(
        cylinder_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR, random_time=USE_RANDOM_TIME
    )
    print_for_check(
        [b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z],
        ["b_cylinder_t", "b_cylinder_x", "b_cylinder_y", "b_cylinder_z"],
    )

    # bc-top nodes discre
    b_top_t, b_top_x, b_top_y, b_top_z = normalized_bc(
        top_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR, random_time=USE_RANDOM_TIME
    )
    print_for_check(
        [b_top_t, b_top_x, b_top_y, b_top_z],
        ["b_top_t", "b_top_x", "b_top_y", "b_top_z"],
    )

    # bc-bottom nodes discre
    b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z = normalized_bc(
        bottom_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR, random_time=USE_RANDOM_TIME
    )
    print_for_check(
        [b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z],
        ["b_bottom_t", "b_bottom_t", "b_bottom_t", "b_bottom_t"],
    )

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = new_psci.equation.NavierStokes(nu=NU, rho=1.0, dim=3, time=True)

    constraint_dict = {}

    int_constraint = new_psci.constraint.InteriorConstraint(
        label_expr=pde.equations,
        label_dict={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom={"t": i_t, "x": i_x, "y": i_y, "z": i_z},
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["interior"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=eq_wgt,
        name="INTERIOR",
    )

    ic_constraint = new_psci.constraint.SupervisedInitialConstraint(
        data_file={
            "init_t": init_t,
            "init_x": init_x,
            "init_y": init_y,
            "init_z": init_z,
            "init_u": init_u,
            "init_v": init_v,
            "init_w": init_w,
            # "init_p": init_p,
        },
        input_keys=["init_t", "init_x", "init_y", "init_z"],
        label_keys=["init_u", "init_v", "init_w"],  # , "init_p"],
        t0=0,
        alias_dict={
            "init_t": "t",
            "init_x": "x",
            "init_y": "y",
            "init_z": "z",
            "init_u": "u",
            "init_v": "v",
            "init_w": "w",
            # "init_p": "p",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["ic"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=ic_wgt,
        name="IC",
    )

    sup_constraint = new_psci.constraint.SupervisedConstraint(
        data_file={
            "sup_t": sup_t,
            "sup_x": sup_x,
            "sup_y": sup_y,
            "sup_z": sup_z,
            "sup_u": sup_u,
            "sup_v": sup_v,
            "sup_w": sup_w,
            # "sup_p": sup_p,
        },
        input_keys=["sup_t", "sup_x", "sup_y", "sup_z"],
        label_keys=["sup_u", "sup_v", "sup_w"],  # , "sup_p"],
        alias_dict={
            "sup_t": "t",
            "sup_x": "x",
            "sup_y": "y",
            "sup_z": "z",
            "sup_u": "u",
            "sup_v": "v",
            "sup_w": "w",
            # "sup_p": "p",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["supervised"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=sup_wgt,
        name="SUP",
    )

    inlet_constraint = new_psci.constraint.SupervisedConstraint(
        data_file={
            "b_inlet_t": b_inlet_t,
            "b_inlet_x": b_inlet_x,
            "b_inlet_y": b_inlet_y,
            "b_inlet_z": b_inlet_z,
            "b_inlet_u": 0.1 / UVW_STAR,
            "b_inlet_v": 0,
            "b_inlet_w": 0,
        },
        input_keys=["b_inlet_t", "b_inlet_x", "b_inlet_y", "b_inlet_z"],
        label_keys=["b_inlet_u", "b_inlet_v", "b_inlet_w"],
        alias_dict={
            "b_inlet_t": "t",
            "b_inlet_x": "x",
            "b_inlet_y": "y",
            "b_inlet_z": "z",
            "b_inlet_u": "u",
            "b_inlet_v": "v",
            "b_inlet_w": "w",
            "b_inlet_p": "p",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["inlet"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=inlet_wgt,
        name="INLET",
    )

    cylinder_constraint = new_psci.constraint.SupervisedConstraint(
        data_file={
            "b_cylinder_t": b_cylinder_t,
            "b_cylinder_x": b_cylinder_x,
            "b_cylinder_y": b_cylinder_y,
            "b_cylinder_z": b_cylinder_z,
            "b_cylinder_u": 0,
            "b_cylinder_v": 0,
            "b_cylinder_w": 0,
        },
        input_keys=["b_cylinder_t", "b_cylinder_x", "b_cylinder_y", "b_cylinder_z"],
        label_keys=["b_cylinder_u", "b_cylinder_v", "b_cylinder_w"],
        alias_dict={
            "b_cylinder_t": "t",
            "b_cylinder_x": "x",
            "b_cylinder_y": "y",
            "b_cylinder_z": "z",
            "b_cylinder_u": "u",
            "b_cylinder_v": "v",
            "b_cylinder_w": "w",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["cylinder"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=cylinder_wgt,
        name="CYLINDER",
    )

    outlet_constraint = new_psci.constraint.SupervisedConstraint(
        data_file={
            "b_outlet_t": b_outlet_t,
            "b_outlet_x": b_outlet_x,
            "b_outlet_y": b_outlet_y,
            "b_outlet_z": b_outlet_z,
            "b_outlet_p": 0,
        },
        input_keys=["b_outlet_t", "b_outlet_x", "b_outlet_y", "b_outlet_z"],
        label_keys=["b_outlet_p"],
        alias_dict={
            "b_outlet_t": "t",
            "b_outlet_x": "x",
            "b_outlet_y": "y",
            "b_outlet_z": "z",
            "b_outlet_p": "p",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["outlet"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=outlet_wgt,
        name="OUTLET",
    )

    top_constraint = new_psci.constraint.SupervisedConstraint(
        data_file={
            "b_top_t": b_top_t,
            "b_top_x": b_top_x,
            "b_top_y": b_top_y,
            "b_top_z": b_top_z,
            "b_top_u": 0.1 / UVW_STAR,
            "b_top_v": 0,
            "b_top_w": 0,
        },
        input_keys=["b_top_t", "b_top_x", "b_top_y", "b_top_z"],
        label_keys=["b_top_u", "b_top_v", "b_top_w"],
        alias_dict={
            "b_top_t": "t",
            "b_top_x": "x",
            "b_top_y": "y",
            "b_top_z": "z",
            "b_top_u": "u",
            "b_top_v": "v",
            "b_top_w": "w",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["top"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=top_wgt,
        name="TOP",
    )

    bottom_constraint = new_psci.constraint.SupervisedConstraint(
        data_file={
            "b_bottom_t": b_bottom_t,
            "b_bottom_x": b_bottom_x,
            "b_bottom_y": b_bottom_y,
            "b_bottom_z": b_bottom_z,
            "b_bottom_u": 0.1 / UVW_STAR,
            "b_bottom_v": 0,
            "b_bottom_w": 0,
        },
        input_keys=["b_bottom_t", "b_bottom_x", "b_bottom_y", "b_bottom_z"],
        label_keys=["b_bottom_u", "b_bottom_v", "b_bottom_w"],
        alias_dict={
            "b_bottom_t": "t",
            "b_bottom_x": "x",
            "b_bottom_y": "y",
            "b_bottom_z": "z",
            "b_bottom_u": "u",
            "b_bottom_v": "v",
            "b_bottom_w": "w",
        },
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 2,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "batch_size": bs["bottom"],
                "drop_last": False,
            },
        },
        loss=new_psci.loss.MSELoss("mean"),
        weight_value=bottom_wgt,
        name="BOTTOM",
    )
    constraint_dict[int_constraint.name] = int_constraint
    constraint_dict[inlet_constraint.name] = inlet_constraint
    constraint_dict[cylinder_constraint.name] = cylinder_constraint
    constraint_dict[outlet_constraint.name] = outlet_constraint
    constraint_dict[top_constraint.name] = top_constraint
    constraint_dict[bottom_constraint.name] = bottom_constraint
    constraint_dict[ic_constraint.name] = ic_constraint
    constraint_dict[sup_constraint.name] = sup_constraint

    model = new_psci.arch.MLP(
        input_keys=["t", "x", "y", "z"],
        output_keys=["u", "v", "w", "p"],
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation_func,
        skip_connection=False,
        weight_norm=False,
    )

    lr = new_psci.optimizer.lr_scheduler.Cosine(
        epochs=num_epoch,
        iters_per_epoch=1,
        learning_rate=learning_rate,
        warmup_epoch=w_epoch,
    )()

    optimizer = new_psci.optimizer.Adam(learning_rate=lr)([model])

    # Read validation reference for time step : 0, 99
    lbm_0_input, lbm_0_output = load_vtk(
        [0],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name,
        load_in_dict_shape=True,
    )
    lbm_99_input, lbm_99_output = load_vtk(
        [99],
        t_step=TIME_STEP,
        load_uvwp=True,
        load_txyz=True,
        name_wt_time=ic_name,
        load_in_dict_shape=True,
    )
    factor_dict = {
        Input.t: T_STAR,
        Input.x: XYZ_STAR,
        Input.y: XYZ_STAR,
        Input.z: XYZ_STAR,
        Output.u: UVW_STAR,
        Output.v: UVW_STAR,
        Output.w: UVW_STAR,
        Output.p: p_star,
    }
    lbm_0_input = normlization(factor_dict, lbm_0_input)
    lbm_0_output = normlization(factor_dict, lbm_0_output)
    lbm_0_old = {}
    lbm_0_old.update(lbm_0_input)
    lbm_0_old.update(lbm_0_output)

    # temporary
    lbm_0 = {}
    for key, value in lbm_0_old.items():
        lbm_0[key.value] = value
    lbm_data_size = len(next(iter(lbm_0.values())))
    # Validator
    validator = {
        "Residual": new_psci.validate.DataValidator(
            data=lbm_0,
            input_keys=["t", "x", "y", "z"],
            label_keys=["u", "v", "w", "p"],
            alias_dict={},
            dataloader_cfg={
                "dataset": "IterableNamedArrayDataset",
                "total_size": lbm_data_size,
                "sampler": None,
            },
            loss=new_psci.loss.MSELoss("mean"),
            metric={"MSE": new_psci.metric.MSE()},
            name="Residual",
        ),
    }

    # Solver
    output_dir = "./outputs_0404/"
    train_solver = new_psci.solver.Solver(
        mode="train",
        model=model,
        constraint=constraint_dict,
        output_dir=output_dir,
        optimizer=optimizer,
        lr_scheduler=lr,
        epochs=num_epoch,
        iters_per_epoch=1,
        save_freq=1000,
        eval_during_train=False,
        eval_freq=1000,
        equation=pde,
        geom=None,
        validator=validator,
        checkpoint_path="./outputs_0404/checkpoints/latest",
    )
    train_solver.train()

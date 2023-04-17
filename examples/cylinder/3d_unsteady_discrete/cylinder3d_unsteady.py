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

import os.path as osp
import sys

import numpy as np

import ppsci
import ppsci.utils.misc as misc


def normalize(input: dict, factor: dict):
    for key, value in factor.items():
        if abs(value) < sys.float_info.min:
            raise ValueError(f"{key} in factor dict is zero")
    normolized_input = {key: value / factor[key] for key, value in input.items()}
    return normolized_input


def denormalize(input: dict, factor: dict):
    denormolized_input = {key: value * factor[key] for key, value in input.items()}
    return denormolized_input


if __name__ == "__main__":
    # set output directory
    output_dir = "./output"
    dirname = "."

    # initialize logger
    ppsci.utils.logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    RENOLDS_NUMBER = 3900
    U0 = 0.1
    D_CYLINDER = 80
    RHO = 1
    NU = RHO * U0 * D_CYLINDER / RENOLDS_NUMBER

    T_STAR = D_CYLINDER / U0  # 800
    XYZ_STAR = D_CYLINDER  # 80
    UVW_STAR = U0  # 0.1
    P_STAR = RHO * U0 * U0  # 0.01

    factor_dict = {
        "t": T_STAR,
        "x": XYZ_STAR,
        "y": XYZ_STAR,
        "z": XYZ_STAR,
        "u": UVW_STAR,
        "v": UVW_STAR,
        "w": UVW_STAR,
        "p": P_STAR,
    }

    num_epoch = 400000  # number of epoch
    learning_rate = 0.001
    hidden_size = 512
    num_layers = 5
    activation_func = "tanh"

    bs = {
        "interior": 4000,
        "inlet": 256,
        "outlet": 256,
        "cylinder": 256,
        "top": 1280,
        "bottom": 1280,
        "ic": 6400,
        "supervised": 6400,
    }  # batch size

    # losses weight
    eq_wgt = 1
    ic_wgt = 5
    front_wgt = 2
    back_wgt = 2
    inlet_wgt = 2
    outlet_wgt = 1
    top_wgt = 2
    bottom_wgt = 2
    cylinder_wgt = 5
    sup_wgt = 10
    w_epoch = 50000  # simulated annealing

    # fix the random seed
    ppsci.utils.misc.set_random_seed(42)

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
    ).astype("int64")
    time_tmp = time_list * TIME_STEP
    time_index = np.random.choice(time_list, int(TIME_NUMBER / 2.5), replace=False)
    time_index.sort()
    time_array = time_index * TIME_STEP

    # initial value
    ref_file = osp.join(dirname, "data/LBM_result/cylinder3d_2023_1_31_LBM_")
    ic_input, ic_label = misc.load_vtk_file(ref_file, TIME_STEP, [0])
    ic_dict = {}
    ic_dict.update(normalize(ic_input, factor_dict))
    ic_dict.update(normalize(ic_label, factor_dict))

    # supervised data
    sup_name = osp.join(dirname, "data/sup_data/supervised_")
    sup_input, sup_label = misc.load_vtk_file(sup_name, TIME_STEP, time_index)
    sup_dict = {}
    sup_dict.update(normalize(sup_input, factor_dict))
    sup_dict.update(normalize(sup_label, factor_dict))

    # interior data
    interior_data = misc.load_vtk_withtime_file(
        osp.join(dirname, "data/sample_points/interior_txyz.vtu")  # input
    )

    # inlet data
    inlet_data = misc.load_vtk_withtime_file(
        osp.join(dirname, "data/sample_points/inlet_txyz.vtu")  # input
    )
    inlet_data.update(
        {
            "u": 0.1,
            "v": 0,
            "w": 0,
        }
    )  # label

    # cylinder data
    cylinder_data = misc.load_vtk_withtime_file(
        osp.join(dirname, "data/sample_points/cylinder_txyz.vtu")  # input
    )
    cylinder_data.update(
        {
            "u": 0,
            "v": 0,
            "w": 0,
        }
    )  # label

    # outlet data
    outlet_data = misc.load_vtk_withtime_file(
        osp.join(dirname, "data/sample_points/outlet_txyz.vtu")  # input
    )
    outlet_data.update(
        {
            "p": 0,
        }
    )  # label

    # top data
    top_data = misc.load_vtk_withtime_file(
        osp.join(dirname, "data/sample_points/top_txyz.vtu")  # input
    )
    top_data.update(
        {
            "u": 0.1,
            "v": 0,
            "w": 0,
        }
    )  # label

    # bottom data
    bottom_data = misc.load_vtk_withtime_file(
        osp.join(dirname, "data/sample_points/bottom_txyz.vtu")  # input
    )
    bottom_data.update(
        {
            "u": 0.1,
            "v": 0,
            "w": 0,
        }
    )  # label

    interior_data = normalize(interior_data, factor_dict)
    bc_inlet_data = normalize(inlet_data, factor_dict)
    bc_cylinder_data = normalize(cylinder_data, factor_dict)
    bc_outlet_data = normalize(outlet_data, factor_dict)
    bc_top_data = normalize(top_data, factor_dict)
    bc_bottom_data = normalize(bottom_data, factor_dict)

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = ppsci.equation.NavierStokes(nu=NU, rho=1.0, dim=3, time=True)
    input_keys = ["t", "x", "y", "z"]
    label_keys_1 = ["u", "v", "w", "p"]
    label_keys_2 = ["u", "v", "w"]
    pde_constraint = ppsci.constraint.InteriorConstraint(
        label_expr=pde.equations,
        label_dict={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom=interior_data,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["interior"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", eq_wgt),
        name="INTERIOR",
    )

    ic = ppsci.constraint.SupervisedConstraint(
        data_file=ic_dict,
        input_keys=input_keys,
        label_keys=label_keys_2,
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["ic"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", ic_wgt),
        name="IC",
    )

    sup = ppsci.constraint.SupervisedConstraint(
        data_file=sup_dict,
        input_keys=input_keys,
        label_keys=label_keys_2,
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["supervised"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", sup_wgt),
        name="SUP",
    )

    bc_inlet = ppsci.constraint.SupervisedConstraint(
        data_file=bc_inlet_data,
        input_keys=input_keys,
        label_keys=label_keys_2,
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["inlet"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", inlet_wgt),
        name="BC_INLET",
    )

    bc_cylinder = ppsci.constraint.SupervisedConstraint(
        data_file=bc_cylinder_data,
        input_keys=input_keys,
        label_keys=label_keys_2,
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["cylinder"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", cylinder_wgt),
        name="BC_CYLINDER",
    )

    bc_outlet = ppsci.constraint.SupervisedConstraint(
        data_file=bc_outlet_data,
        input_keys=input_keys,
        label_keys=["p"],
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["outlet"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", outlet_wgt),
        name="BC_OUTLET",
    )

    bc_top = ppsci.constraint.SupervisedConstraint(
        data_file=bc_top_data,
        input_keys=input_keys,
        label_keys=label_keys_2,
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["top"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", top_wgt),
        name="BC_TOP",
    )

    bc_bottom = ppsci.constraint.SupervisedConstraint(
        data_file=bc_bottom_data,
        input_keys=input_keys,
        label_keys=label_keys_2,
        alias_dict=None,
        dataloader_cfg={
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": bs["bottom"],
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", bottom_wgt),
        name="BC_BOTTOM",
    )

    constraint_dict = {
        pde_constraint.name: pde_constraint,
        bc_inlet.name: bc_inlet,
        bc_cylinder.name: bc_cylinder,
        bc_outlet.name: bc_outlet,
        bc_top.name: bc_top,
        bc_bottom.name: bc_bottom,
        ic.name: ic,
        sup.name: sup,
    }

    model = ppsci.arch.MLP(
        input_keys=["t", "x", "y", "z"],
        output_keys=label_keys_1,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation_func,
        skip_connection=False,
        weight_norm=False,
    )

    lr = ppsci.optimizer.lr_scheduler.Cosine(
        epochs=num_epoch,
        iters_per_epoch=1,
        learning_rate=learning_rate,
        warmup_epoch=w_epoch,
    )()

    optimizer = ppsci.optimizer.Adam(learning_rate=lr)([model])

    # Read validation reference for time step : 0, 99

    lbm_0_input, lbm_0_label = misc.load_vtk_file(ref_file, TIME_STEP, [0])
    lbm_99_input, lbm_99_label = misc.load_vtk_file(ref_file, TIME_STEP, [99])

    lbm_0_input = normalize(lbm_0_input, factor_dict)
    lbm_0_label = normalize(lbm_0_label, factor_dict)
    lbm_0_dict = {}
    lbm_0_dict.update(lbm_0_input)
    lbm_0_dict.update(lbm_0_label)

    # Validator
    validator = {
        "Residual": ppsci.validate.DataValidator(
            data_dict=lbm_0_dict,
            input_keys=input_keys,
            label_keys=label_keys_1,
            alias_dict={},
            dataloader_cfg={
                "dataset": "NamedArrayDataset",
                "total_size": len(next(iter(lbm_0_dict.values()))),
                "batch_size": 1024,
                "sampler": {"name": "BatchSampler"},
            },
            loss=ppsci.loss.MSELoss("mean"),
            metric={"MSE": ppsci.metric.MSE()},
            name="Residual",
        ),
    }
    pretrained_model_path = osp.join(dirname, "checkpoints/epoch_301000")
    one_input, _ = misc.load_vtk_file(ref_file, TIME_STEP, [0])
    cord = {"x": one_input["x"], "y": one_input["y"], "z": one_input["z"]}

    visualizer = {
        "visulzie_uvwp": ppsci.visualize.Visualizer3D(
            time_step=TIME_STEP,
            time_list=time_list,
            factor_dict=factor_dict,
            input_dict=cord,
            output_expr={
                "u": lambda d: d["u"],
                "v": lambda d: d["v"],
                "w": lambda d: d["w"],
                "p": lambda d: d["p"],
            },
            ref_file=ref_file,
            prefix="result_uvwp",
        )
    }

    # Solver
    train_solver = ppsci.solver.Solver(
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
        visualizer=visualizer,
    )
    train_solver.train()

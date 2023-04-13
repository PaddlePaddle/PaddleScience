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
import os.path as osp
import sys

import numpy as np

import ppsci
import ppsci.data.dataset as dataset


def normalized_bc(origion_list, t_factor, xyz_factor):
    """normalize bc data time and coordinates

    Args:
        origion_list (np.array): Array to be normalized
        t_factor (float): Scaling factor for time
        xyz_factor (float): Scaling factor for x y z

    Returns:
        np.array: Normalized array
    """
    time = origion_list[:, 0] / t_factor
    x_cord = origion_list[:, 1] / xyz_factor
    y_cord = origion_list[:, 2] / xyz_factor
    z_cord = origion_list[:, 3] / xyz_factor
    return time, x_cord, y_cord, z_cord


if __name__ == "__main__":
    # set output directory
    output_dir = "./outputs_0404/"
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
        dataset.Input.t: T_STAR,
        dataset.Input.x: XYZ_STAR,
        dataset.Input.y: XYZ_STAR,
        dataset.Input.z: XYZ_STAR,
        dataset.Label.u: UVW_STAR,
        dataset.Label.v: UVW_STAR,
        dataset.Label.w: UVW_STAR,
        dataset.Label.p: P_STAR,
    }

    # read configuration file : config.yaml
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    # dirname = "."

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
    ic_wgt = 1
    eq_wgt = 1
    front_wgt = 2
    back_wgt = 2
    inlet_wgt = 2
    outlet_wgt = 1
    top_wgt = 2
    bottom_wgt = 2
    cylinder_wgt = 5
    sup_wgt = 10
    w_epoch = 5000  # simulated annealing

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

    # Data reader
    reader = dataset.Reader(time_index=time_index, time_step=TIME_STEP)

    # initial value
    ref_file = osp.join(dirname, "data/LBM_result/cylinder3d_2023_1_31_LBM_")
    ic_name = ref_file
    ic_input, ic_label = reader.vtk(time_point=0, filename_without_timeid=ic_name)

    ic_dict = {}
    ic_dict.update(dataset.normalization(ic_input, factor_dict))
    ic_dict.update(dataset.normalization(ic_label, factor_dict))

    # supervised data
    sup_name = osp.join(dirname, "data/sup_data/supervised_")
    sup_input, sup_label = reader.vtk(filename_without_timeid=sup_name)

    sup_dict = {}
    sup_dict.update(dataset.normalization(sup_input, factor_dict))
    sup_dict.update(dataset.normalization(sup_label, factor_dict))

    # num of interior points
    NUM_POINTS = 10000
    inlet_txyz = reader.vtk_samples_with_time(
        osp.join(dirname, "data/sample_points/inlet_txyz.vtu")
    )
    outlet_txyz = reader.vtk_samples_with_time(
        osp.join(dirname, "data/sample_points/outlet_txyz.vtu")
    )
    top_txyz = reader.vtk_samples_with_time(
        osp.join(dirname, "data/sample_points/top_txyz.vtu")
    )
    bottom_txyz = reader.vtk_samples_with_time(
        osp.join(dirname, "data/sample_points/bottom_txyz.vtu")
    )
    cylinder_txyz = reader.vtk_samples_with_time(
        osp.join(dirname, "data/sample_points/cylinder_txyz.vtu")
    )
    interior_txyz = reader.vtk_samples_with_time(
        osp.join(dirname, "data/sample_points/interior_txyz.vtu")
    )

    # interior nodes discre
    i_t, i_x, i_y, i_z = normalized_bc(
        interior_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR
    )

    # bc inlet nodes discre
    b_inlet_t, b_inlet_x, b_inlet_y, b_inlet_z = normalized_bc(
        inlet_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR
    )

    # bc outlet nodes discre
    b_outlet_t, b_outlet_x, b_outlet_y, b_outlet_z = normalized_bc(
        outlet_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR
    )

    # bc cylinder nodes discre
    b_cylinder_t, b_cylinder_x, b_cylinder_y, b_cylinder_z = normalized_bc(
        cylinder_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR
    )

    # bc-top nodes discre
    b_top_t, b_top_x, b_top_y, b_top_z = normalized_bc(
        top_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR
    )

    # bc-bottom nodes discre
    b_bottom_t, b_bottom_x, b_bottom_y, b_bottom_z = normalized_bc(
        bottom_txyz, t_factor=T_STAR, xyz_factor=XYZ_STAR
    )

    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    pde = ppsci.equation.NavierStokes(nu=NU, rho=1.0, dim=3, time=True)
    input_keys = [dataset.Input.t, dataset.Input.x, dataset.Input.y, dataset.Input.z]
    label_keys_1 = [dataset.Label.u, dataset.Label.v, dataset.Label.w, dataset.Label.p]
    label_keys_2 = [dataset.Label.u, dataset.Label.v, dataset.Label.w]
    pde_constraint = ppsci.constraint.InteriorConstraint(
        label_expr=pde.equations,
        label_dict={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom={
            dataset.Input.t: i_t,
            dataset.Input.x: i_x,
            dataset.Input.y: i_y,
            dataset.Input.z: i_z,
        },
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=eq_wgt,
        name="INTERIOR",
    )

    ic = ppsci.constraint.SupervisedInitialConstraint(
        data_file=ic_dict,
        input_keys=input_keys,
        label_keys=label_keys_2,
        t0=0,
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=ic_wgt,
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=sup_wgt,
        name="SUP",
    )

    bc_inlet = ppsci.constraint.SupervisedConstraint(
        data_file={
            dataset.Input.t: b_inlet_t,
            dataset.Input.x: b_inlet_x,
            dataset.Input.y: b_inlet_y,
            dataset.Input.z: b_inlet_z,
            dataset.Label.u: 0.1 / UVW_STAR,
            dataset.Label.v: 0,
            dataset.Label.w: 0,
        },
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=inlet_wgt,
        name="BC_INLET",
    )

    bc_cylinder = ppsci.constraint.SupervisedConstraint(
        data_file={
            dataset.Input.t: b_cylinder_t,
            dataset.Input.x: b_cylinder_x,
            dataset.Input.y: b_cylinder_y,
            dataset.Input.z: b_cylinder_z,
            dataset.Label.u: 0,
            dataset.Label.v: 0,
            dataset.Label.w: 0,
        },
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=cylinder_wgt,
        name="BC_CYLINDER",
    )

    bc_outlet = ppsci.constraint.SupervisedConstraint(
        data_file={
            dataset.Input.t: b_outlet_t,
            dataset.Input.x: b_outlet_x,
            dataset.Input.y: b_outlet_y,
            dataset.Input.z: b_outlet_z,
            dataset.Label.p: 0,
        },
        input_keys=input_keys,
        label_keys=[dataset.Label.p],
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=outlet_wgt,
        name="BC_OUTLET",
    )

    bc_top = ppsci.constraint.SupervisedConstraint(
        data_file={
            dataset.Input.t: b_top_t,
            dataset.Input.x: b_top_x,
            dataset.Input.y: b_top_y,
            dataset.Input.z: b_top_z,
            dataset.Label.u: 0.1 / UVW_STAR,
            dataset.Label.v: 0,
            dataset.Label.w: 0,
        },
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=top_wgt,
        name="BC_TOP",
    )

    bc_bottom = ppsci.constraint.SupervisedConstraint(
        data_file={
            dataset.Input.t: b_bottom_t,
            dataset.Input.x: b_bottom_x,
            dataset.Input.y: b_bottom_y,
            dataset.Input.z: b_bottom_z,
            dataset.Label.u: 0.1 / UVW_STAR,
            dataset.Label.v: 0,
            dataset.Label.w: 0,
        },
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
        loss=ppsci.loss.MSELoss("mean"),
        weight_value=bottom_wgt,
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
        input_keys=[dataset.Input.t, dataset.Input.x, dataset.Input.y, dataset.Input.z],
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
    lbm_0_input, lbm_0_label = reader.vtk(
        time_point=0, filename_without_timeid=ref_file
    )
    lbm_99_input, lbm_99_label = reader.vtk(
        time_point=99, filename_without_timeid=ref_file
    )
    lbm_0_input = dataset.normalization(lbm_0_input, factor_dict)
    lbm_0_label = dataset.normalization(lbm_0_label, factor_dict)
    lbm_0_dict = {}
    lbm_0_dict.update(lbm_0_input)
    lbm_0_dict.update(lbm_0_label)

    # Validator
    validator = {
        "Residual": ppsci.validate.DataValidator(
            data=lbm_0_dict,
            input_keys=input_keys,
            label_keys=label_keys_1,
            alias_dict={},
            dataloader_cfg={
                "dataset": "NamedArrayDataset",
                "total_size": len(next(iter(lbm_0_dict.values()))),
                "batch_size": bs["bottom"],
                "sampler": {
                    "name": "BatchSampler",
                    "shuffle": False,
                    "drop_last": False,
                },
            },
            loss=ppsci.loss.MSELoss("mean"),
            metric={"MSE": ppsci.metric.MSE()},
            name="Residual",
        ),
    }
    pretrained_model_path = osp.join(dirname, "checkpoints/epoch_301000")
    one_input, _ = reader.vtk(time_point=0, filename_without_timeid=ref_file)
    cord = {
        dataset.Input.x: one_input[dataset.Input.x],
        dataset.Input.y: one_input[dataset.Input.y],
        dataset.Input.z: one_input[dataset.Input.z],
    }

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
    )
    train_solver.train()

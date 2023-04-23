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

import numpy as np

import ppsci
import ppsci.data.process.transform as transform
import ppsci.utils.reader as reader
from ppsci.utils import logger

if __name__ == "__main__":

    import time

    t_start = time.time()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    import os

    os.chdir("/workspace/wangguan/PaddleScience/examples/cylinder/3d_unsteady_discrete")
    # set output directory
    output_dir = "./output_debug"

    ref_file = "data/LBM_result/cylinder3d_2023_1_31_LBM_"

    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(
        ("t", "x", "y", "z"),
        ("u", "v", "w", "p"),
        5,
        512,
    )

    # set equation and necessary constant
    RENOLDS_NUMBER = 3900
    U0 = 0.1
    D_CYLINDER = 80
    RHO = 1
    NU = RHO * U0 * D_CYLINDER / RENOLDS_NUMBER

    T_STAR = D_CYLINDER / U0  # 800
    XYZ_STAR = D_CYLINDER  # 80
    UVW_STAR = U0  # 0.1
    P_STAR = RHO * U0 * U0  # 0.01
    # N-S, Re=3900, D=80, u=0.1, nu=80/3900; nu = rho u D / Re = 1.0 * 0.1 * 80 / 3900
    equation = {"NavierStokes": ppsci.equation.NavierStokes(NU, RHO, 3, True)}

    # set geometry
    norm_factor = {
        "t": T_STAR,
        "x": XYZ_STAR,
        "y": XYZ_STAR,
        "z": XYZ_STAR,
        "u": UVW_STAR,
        "v": UVW_STAR,
        "w": UVW_STAR,
        "p": P_STAR,
    }
    normalize = transform.Scale({key: 1 / value for key, value in norm_factor.items()})
    interior_data = reader.load_vtk_withtime_file(
        "data/sample_points/interior_txyz.vtu"
    )
    geom = {
        "interior": ppsci.geometry.PointCloud(
            interior=normalize(interior_data),
            attributes=None,
            data_key=("t", "x", "y", "z"),
        )
    }

    # set dataloader config
    batchsize_interior = 4000
    batchsize_inlet = 256
    batchsize_outlet = 256
    batchsize_cylinder = 256
    batchsize_top = 1280
    batchsize_bottom = 1280
    batchsize_ic = 6400
    batchsize_supervised = 6400

    # set time array
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

    input_keys = ["t", "x", "y", "z"]
    label_keys = ["u", "v", "w", "p"]

    label_expr = {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]}
    label_expr_2 = {
        "u": lambda d: d["u"],
        "v": lambda d: d["v"],
        "w": lambda d: d["w"],
        "p": lambda d: d["p"],
    }

    # set constraint
    _normalize = {
        "Scale": {"scale": {key: 1 / value for key, value in norm_factor.items()}}
    }
    # interior data
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom["interior"],
        evenly=True,
        dataloader_cfg={
            "iters_per_epoch": int(geom["interior"].len / batchsize_interior),
            "dataset": "MiniBatchDataset",
            "num_workers": 1,
            "batch_size": batchsize_interior,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean", 1),
        name="INTERIOR",
    )

    ic = ppsci.constraint.SupervisedConstraint(
        label_expr=label_expr,
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": ref_file,
                "input_keys": input_keys,
                "label_keys": ("u", "v", "w"),
                "time_step": TIME_STEP,
                "time_index": [0],
                "transforms": [_normalize],
            },
            "batch_size": batchsize_ic,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "num_workers": 1,
        },
        loss=ppsci.loss.MSELoss("mean", 5),
        name="IC",
    )

    sup = ppsci.constraint.SupervisedConstraint(
        label_expr=label_expr,
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": "data/sup_data/supervised_",
                "input_keys": input_keys,
                "label_keys": ("u", "v", "w"),
                "time_step": TIME_STEP,
                "time_index": time_index,
                "transforms": [_normalize],
            },
            "batch_size": batchsize_supervised,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "num_workers": 1,
        },
        loss=ppsci.loss.MSELoss("mean", 10),
        name="SUP",
    )

    bc_inlet = ppsci.constraint.SupervisedConstraint(
        label_expr=label_expr,
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": "data/sample_points/inlet_txyz.vtu",
                "input_keys": input_keys,
                "label_keys": ("u", "v", "w"),
                "labels": {"u": 0.1, "v": 0, "w": 0},
                "transforms": [_normalize],
            },
            "batch_size": batchsize_inlet,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "num_workers": 1,
        },
        loss=ppsci.loss.MSELoss("mean", 2),
        name="BC_INLET",
    )

    bc_cylinder = ppsci.constraint.SupervisedConstraint(
        label_expr=label_expr,
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": "data/sample_points/cylinder_txyz.vtu",
                "input_keys": input_keys,
                "label_keys": ("u", "v", "w"),
                "labels": {"u": 0, "v": 0, "w": 0},
                "transforms": [_normalize],
            },
            "num_workers": 1,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "batch_size": batchsize_cylinder,
        },
        loss=ppsci.loss.MSELoss("mean", 5),
        name="BC_CYLINDER",
    )

    bc_outlet = ppsci.constraint.SupervisedConstraint(
        label_expr={"p": lambda d: d["p"]},
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": "data/sample_points/outlet_txyz.vtu",
                "input_keys": input_keys,
                "label_keys": ["p"],
                "labels": {"p": 0},
                "transforms": [_normalize],
            },
            "batch_size": batchsize_outlet,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "num_workers": 1,
        },
        loss=ppsci.loss.MSELoss("mean", 1),
        name="BC_OUTLET",
    )

    bc_top = ppsci.constraint.SupervisedConstraint(
        label_expr=label_expr,
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": "data/sample_points/top_txyz.vtu",
                "input_keys": input_keys,
                "label_keys": ("u", "v", "w"),
                "labels": {"u": 0.1, "v": 0, "w": 0},
                "transforms": [_normalize],
            },
            "batch_size": batchsize_top,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "num_workers": 1,
        },
        loss=ppsci.loss.MSELoss("mean", 2),
        name="BC_TOP",
    )

    bc_bottom = ppsci.constraint.SupervisedConstraint(
        label_expr=label_expr,
        dataloader_cfg={
            "dataset": {
                "name": "VtuDataset",
                "file_path": "data/sample_points/bottom_txyz.vtu",
                "input_keys": input_keys,
                "label_keys": ("u", "v", "w"),
                "labels": {"u": 0.1, "v": 0, "w": 0},
                "transforms": [_normalize],
            },
            "batch_size": batchsize_bottom,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
            "num_workers": 1,
        },
        loss=ppsci.loss.MSELoss("mean", 2),
        name="BC_BOTTOM",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_inlet.name: bc_inlet,
        bc_cylinder.name: bc_cylinder,
        bc_outlet.name: bc_outlet,
        bc_top.name: bc_top,
        bc_bottom.name: bc_bottom,
        ic.name: ic,
        sup.name: sup,
    }

    # set training hyper-parameters
    epochs = 400000
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        epochs=epochs,
        iters_per_epoch=1,
        learning_rate=0.001,
        warmup_epoch=int(epochs * 0.125),
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=lr_scheduler)((model,))

    # Read validation reference for time step : 0, 99
    lbm_0_input, lbm_0_label = reader.load_vtk_file(
        ref_file, TIME_STEP, [0], input_keys, label_keys
    )
    lbm_99_input, lbm_99_label = reader.load_vtk_file(
        ref_file, TIME_STEP, [99], input_keys, label_keys
    )

    lbm_0_input = normalize(lbm_0_input)
    lbm_0_label = normalize(lbm_0_label)
    lbm_0_dict = {**lbm_0_input, **lbm_0_label}

    # set visualizer(optional)
    validator = {
        "Residual": ppsci.validate.SupervisedValidator(
            dataloader_cfg={
                "dataset": {
                    "name": "VtuDataset",
                    "file_path": ref_file,
                    "input_keys": input_keys,
                    "label_keys": ("u", "v", "w"),
                    "time_step": TIME_STEP,
                    "time_index": [0],
                    "transforms": [_normalize],
                },
                "total_size": len(next(iter(lbm_0_dict.values()))),
                "batch_size": 1024,
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

    # interior+boundary
    # manually collate input data for visualization,
    denormalize = transform.Scale(norm_factor)
    visualizer = {
        "visulzie_uvwp": ppsci.visualize.Visualizer3D(
            time_step=TIME_STEP,
            time_list=time_list,
            input_dict={key: None for key in input_keys},
            output_expr={
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
                "p": lambda out: out["p"],
            },
            ref_file=ref_file,
            transforms={"denormalize": denormalize, "normalize": normalize},
            prefix="result_uvwp",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        optimizer,
        lr_scheduler,
        epochs,
        1,
        save_freq=1000,
        eval_during_train=False,
        eval_freq=1000,
        equation=equation,
        geom=None,
        validator=validator,
        visualizer=visualizer,
    )
    t_end = time.time()
    print(f"time cost {t_end - t_start}")
    # train model
    solver.train()

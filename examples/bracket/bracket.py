# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html
STL data files download link: https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
pretrained model download link: https://paddle-org.bj.bcebos.com/paddlescience/models/bracket/bracket_pretrained.pdparams
"""

import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    # set output directory
    OUTPUT_DIR = "./output_bracket" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    disp_net = ppsci.arch.MLP(
        ("x", "y", "z"), ("u", "v", "w"), 6, 512, "silu", weight_norm=True
    )
    stress_net = ppsci.arch.MLP(
        ("x", "y", "z"),
        ("sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"),
        6,
        512,
        "silu",
        weight_norm=True,
    )
    # wrap to a model_list
    model = ppsci.arch.ModelList((disp_net, stress_net))

    # Specify parameters
    NU = 0.3
    E = 100.0e9
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))
    MU = E / (2 * (1 + NU))
    MU_C = 0.01 * MU
    LAMBDA_ = LAMBDA_ / MU_C
    MU = MU / MU_C
    CHARACTERISTIC_LENGTH = 1.0
    CHARACTERISTIC_DISPLACEMENT = 1.0e-4
    SIGMA_NORMALIZATION = CHARACTERISTIC_LENGTH / (CHARACTERISTIC_DISPLACEMENT * MU_C)
    T = -4.0e4 * SIGMA_NORMALIZATION

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
        )
    }

    # set geometry
    support = ppsci.geometry.Mesh("./stl/support.stl")
    bracket = ppsci.geometry.Mesh("./stl/bracket.stl")
    aux_lower = ppsci.geometry.Mesh("./stl/aux_lower.stl")
    aux_upper = ppsci.geometry.Mesh("./stl/aux_upper.stl")
    cylinder_hole = ppsci.geometry.Mesh("./stl/cylinder_hole.stl")
    cylinder_lower = ppsci.geometry.Mesh("./stl/cylinder_lower.stl")
    cylinder_upper = ppsci.geometry.Mesh("./stl/cylinder_upper.stl")
    # geometry bool operation
    curve_lower = aux_lower - cylinder_lower
    curve_upper = aux_upper - cylinder_upper
    geo = support + bracket + curve_lower + curve_upper - cylinder_hole

    geom = {"geo": geo}

    # set dataloader config
    ITERS_PER_EPOCH = 1000
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    SUPPORT_ORIGIN = (-1, -1, -1)
    BRACKET_ORIGIN = (-0.75, -1, -0.1)
    BRACKET_DIM = (1.75, 2, 0.2)
    BOUNDS_SUPPORT_X = (-1, -0.65)
    BOUNDS_SUPPORT_Y = (-1, 1)
    BOUNDS_SUPPORT_Z = (-1, 1)
    BOUNDS_BRACKET_X = (-0.65, 1)
    BOUNDS_BRACKET_Y = (-1, 1)
    BOUNDS_BRACKET_Z = (-0.1, 0.1)

    bc_back = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 1024},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: x == SUPPORT_ORIGIN[0],
        weight_dict={"u": 10, "v": 10, "w": 10},
        name="BC_BACK",
    )
    bc_front = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": T},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: x == BRACKET_ORIGIN[0] + BRACKET_DIM[0],
        name="BC_FRONT",
    )
    bc_surface = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 4096},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.logical_and(
            x > SUPPORT_ORIGIN[0] + 1e-7, x < BRACKET_ORIGIN[0] + BRACKET_DIM[0] - 1e-7
        ),
        name="BC_SURFACE",
    )
    support_interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "equilibrium_z": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 2048},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_SUPPORT_X[0] < x)
            & (x < BOUNDS_SUPPORT_X[1])
            & (BOUNDS_SUPPORT_Y[0] < y)
            & (y < BOUNDS_SUPPORT_Y[1])
            & (BOUNDS_SUPPORT_Z[0] < z)
            & (z < BOUNDS_SUPPORT_Z[1])
        ),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "equilibrium_z": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_zz": "sdf",
            "stress_disp_xy": "sdf",
            "stress_disp_xz": "sdf",
            "stress_disp_yz": "sdf",
        },
        name="support_interior",
    )
    bracket_interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "equilibrium_z": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 1024},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_BRACKET_X[0] < x)
            & (x < BOUNDS_BRACKET_X[1])
            & (BOUNDS_BRACKET_Y[0] < y)
            & (y < BOUNDS_BRACKET_Y[1])
            & (BOUNDS_BRACKET_Z[0] < z)
            & (z < BOUNDS_BRACKET_Z[1])
        ),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "equilibrium_z": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_zz": "sdf",
            "stress_disp_xy": "sdf",
            "stress_disp_xz": "sdf",
            "stress_disp_yz": "sdf",
        },
        name="bracket_interior",
    )
    # wrap constraints together
    constraint = {
        bc_back.name: bc_back,
        bc_front.name: bc_front,
        bc_surface.name: bc_surface,
        support_interior_constraint.name: support_interior_constraint,
        bracket_interior_constraint.name: bracket_interior_constraint,
    }

    # set training hyper-parameters
    EPOCHS = 2000 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        0.95,
        15000,
        by_epoch=False,
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # set validator
    ref_xyzu = ppsci.utils.reader.load_csv_file(
        "./data/deformation_x.txt",
        ("x", "y", "z", "u"),
        {
            "x": "X Location (m)",
            "y": "Y Location (m)",
            "z": "Z Location (m)",
            "u": "Directional Deformation (m)",
        },
        "\t",
    )
    ref_v = ppsci.utils.reader.load_csv_file(
        "./data/deformation_y.txt",
        ("v",),
        {"v": "Directional Deformation (m)"},
        "\t",
    )
    ref_w = ppsci.utils.reader.load_csv_file(
        "./data/deformation_z.txt",
        ("w",),
        {"w": "Directional Deformation (m)"},
        "\t",
    )

    ref_sxx = ppsci.utils.reader.load_csv_file(
        "./data/normal_x.txt",
        ("sigma_xx",),
        {"sigma_xx": "Normal Stress (Pa)"},
        "\t",
    )
    ref_syy = ppsci.utils.reader.load_csv_file(
        "./data/normal_y.txt",
        ("sigma_yy",),
        {"sigma_yy": "Normal Stress (Pa)"},
        "\t",
    )
    ref_szz = ppsci.utils.reader.load_csv_file(
        "./data/normal_z.txt",
        ("sigma_zz",),
        {"sigma_zz": "Normal Stress (Pa)"},
        "\t",
    )

    ref_sxy = ppsci.utils.reader.load_csv_file(
        "./data/shear_xy.txt",
        ("sigma_xy",),
        {"sigma_xy": "Shear Stress (Pa)"},
        "\t",
    )
    ref_sxz = ppsci.utils.reader.load_csv_file(
        "./data/shear_xz.txt",
        ("sigma_xz",),
        {"sigma_xz": "Shear Stress (Pa)"},
        "\t",
    )
    ref_syz = ppsci.utils.reader.load_csv_file(
        "./data/shear_yz.txt",
        ("sigma_yz",),
        {"sigma_yz": "Shear Stress (Pa)"},
        "\t",
    )

    input_dict = {
        "x": ref_xyzu["x"],
        "y": ref_xyzu["y"],
        "z": ref_xyzu["z"],
    }
    label_dict = {
        "u": ref_xyzu["u"] / CHARACTERISTIC_DISPLACEMENT,
        "v": ref_v["v"] / CHARACTERISTIC_DISPLACEMENT,
        "w": ref_w["w"] / CHARACTERISTIC_DISPLACEMENT,
        "sigma_xx": ref_sxx["sigma_xx"] * SIGMA_NORMALIZATION,
        "sigma_yy": ref_syy["sigma_yy"] * SIGMA_NORMALIZATION,
        "sigma_zz": ref_szz["sigma_zz"] * SIGMA_NORMALIZATION,
        "sigma_xy": ref_sxy["sigma_xy"] * SIGMA_NORMALIZATION,
        "sigma_xz": ref_sxz["sigma_xz"] * SIGMA_NORMALIZATION,
        "sigma_yz": ref_syz["sigma_yz"] * SIGMA_NORMALIZATION,
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
            "weight": {k: np.ones_like(v) for k, v in label_dict.items()},
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss("mean"),
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
            "sigma_xx": lambda out: out["sigma_xx"],
            "sigma_yy": lambda out: out["sigma_yy"],
            "sigma_zz": lambda out: out["sigma_zz"],
            "sigma_xy": lambda out: out["sigma_xy"],
            "sigma_xz": lambda out: out["sigma_xz"],
            "sigma_yz": lambda out: out["sigma_yz"],
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="commercial_ref_u_v_w_sigmas",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visulzie_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
                "sigma_xx": lambda out: out["sigma_xx"],
                "sigma_yy": lambda out: out["sigma_yy"],
                "sigma_zz": lambda out: out["sigma_zz"],
                "sigma_xy": lambda out: out["sigma_xy"],
                "sigma_xz": lambda out: out["sigma_xz"],
                "sigma_yz": lambda out: out["sigma_yz"],
            },
            prefix="result_u_v_w_sigmas",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=20,
        log_freq=20,
        eval_during_train=True,
        eval_freq=20,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate pretrained model(optional)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/best_model",
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()

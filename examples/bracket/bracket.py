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

import numpy as np
import paddle
from paddle import fluid

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    # enable prim_eager mode for high-order autodiff
    fluid.core.set_prim_eager_enabled(True)

    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    output_dir = "./output_bracket" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

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

    model.load_dict(
        paddle.load(
            "/workspace/hesensen/PaddleScience_docs/examples/bracket/converted_bracket_initial_ckpt.pdparams"
        )
    )
    logger.info("load pytorch's init weight")

    rand_input = {
        "x": paddle.to_tensor(
            np.load(
                "/workspace/hesensen/modulus/examples/bracket/outputs/bracket/x.npy"
            ),
            stop_gradient=False,
        ),
        "y": paddle.to_tensor(
            np.load(
                "/workspace/hesensen/modulus/examples/bracket/outputs/bracket/y.npy"
            ),
            stop_gradient=False,
        ),
        "z": paddle.to_tensor(
            np.load(
                "/workspace/hesensen/modulus/examples/bracket/outputs/bracket/z.npy"
            ),
            stop_gradient=False,
        ),
    }
    output_disp = disp_net(rand_input)
    for k, v in output_disp.items():
        print(
            f"disp_net: {k} {tuple(v.shape)} {v.min().item():.10f} {v.max().item():.10f} {v.mean().item():.10f} {v.std().item():.10f}"
        )

    output_stress = stress_net(rand_input)
    for k, v in output_stress.items():
        print(
            f"stress_net: {k} {tuple(v.shape)} {v.min().item():.10f} {v.max().item():.10f} {v.mean().item():.10f} {v.std().item():.10f}"
        )
    exit()

    # Specify parameters
    nu = 0.3
    E = 100.0e9
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    mu_c = 0.01 * mu
    lambda_ = lambda_ / mu_c
    mu = mu / mu_c
    characteristic_length = 1.0
    characteristic_displacement = 1.0e-4
    sigma_normalization = characteristic_length / (characteristic_displacement * mu_c)
    T = -4.0e4 * sigma_normalization

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=lambda_, mu=mu, dim=3
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

    curve_lower = aux_lower - cylinder_lower
    curve_upper = aux_upper - cylinder_upper
    geo = support + bracket + curve_lower + curve_upper - cylinder_hole

    geom = {"geo": geo}

    # set dataloader config
    iters_per_epoch = 1000
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }

    support_origin = (-1, -1, -1)
    support_dim = (0.25, 2, 2)
    bracket_origin = (-0.75, -1, -0.1)
    bracket_dim = (1.75, 2, 0.2)
    cylinder_radius = 0.1
    cylinder_height = 2.0
    aux_lower_origin = (-0.75, -1, -0.1 - cylinder_radius)
    aux_lower_dim = (cylinder_radius, 2, cylinder_radius)
    aux_upper_origin = (-0.75, -1, 0.1)
    aux_upper_dim = (cylinder_radius, 2, cylinder_radius)
    cylinder_lower_center = (-0.75 + cylinder_radius, 0, 0)
    cylinder_upper_center = (-0.75 + cylinder_radius, 0, 0)
    cylinder_hole_radius = 0.7
    cylinder_hole_height = 0.5
    cylinder_hole_center = (0.125, 0, 0)

    # set constraint
    bc_back = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 1, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 1024},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.isclose(x, support_origin[0]),
        weight_dict={"u": 10, "v": 10, "w": 10},
        name="BC_BACK",
    )
    bc_front = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": T},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.isclose(x, bracket_origin[0] + bracket_dim[0]),
        name="BC_FRONT",
    )
    bc_surface = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 4096},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.logical_and(
            x > support_origin[0], x < bracket_origin[0] + bracket_dim[0]
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
        # bounds={x: bounds_bracket_x, y: bounds_bracket_y, z: bounds_bracket_z}
        # lambda_weighting={
        #     "equilibrium_x": Symbol("sdf"),
        #     "equilibrium_y": Symbol("sdf"),
        #     "equilibrium_z": Symbol("sdf"),
        #     "stress_disp_xx": Symbol("sdf"),
        #     "stress_disp_yy": Symbol("sdf"),
        #     "stress_disp_zz": Symbol("sdf"),
        #     "stress_disp_xy": Symbol("sdf"),
        #     "stress_disp_xz": Symbol("sdf"),
        #     "stress_disp_yz": Symbol("sdf"),
        # }
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
        {**train_dataloader_cfg, "batch_size": 2048},
        ppsci.loss.MSELoss("sum"),
        # bounds={x: bounds_bracket_x, y: bounds_bracket_y, z: bounds_bracket_z}
        # lambda_weighting={
        #     "equilibrium_x": Symbol("sdf"),
        #     "equilibrium_y": Symbol("sdf"),
        #     "equilibrium_z": Symbol("sdf"),
        #     "stress_disp_xx": Symbol("sdf"),
        #     "stress_disp_yy": Symbol("sdf"),
        #     "stress_disp_zz": Symbol("sdf"),
        #     "stress_disp_xy": Symbol("sdf"),
        #     "stress_disp_xz": Symbol("sdf"),
        #     "stress_disp_yz": Symbol("sdf"),
        # }
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
    epochs = 2000 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        epochs,
        iters_per_epoch,
        1.0e-3,
        0.95,
        15000,
        by_epoch=False,
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # set validator
    std_xyzu = ppsci.utils.reader.load_csv_file(
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
    std_v = ppsci.utils.reader.load_csv_file(
        "./data/deformation_y.txt",
        ("v",),
        {"v": "Directional Deformation (m)"},
        "\t",
    )
    std_w = ppsci.utils.reader.load_csv_file(
        "./data/deformation_z.txt",
        ("w",),
        {"w": "Directional Deformation (m)"},
        "\t",
    )

    std_sxx = ppsci.utils.reader.load_csv_file(
        "./data/normal_y.txt",
        ("sigma_xx",),
        {"sigma_xx": "Normal Stress (Pa)"},
        "\t",
    )
    std_syy = ppsci.utils.reader.load_csv_file(
        "./data/normal_y.txt",
        ("sigma_yy",),
        {"sigma_yy": "Normal Stress (Pa)"},
        "\t",
    )
    std_szz = ppsci.utils.reader.load_csv_file(
        "./data/normal_z.txt",
        ("sigma_zz",),
        {"sigma_zz": "Normal Stress (Pa)"},
        "\t",
    )

    std_sxy = ppsci.utils.reader.load_csv_file(
        "./data/shear_xy.txt",
        ("sigma_xy",),
        {"sigma_xy": "Shear Stress (Pa)"},
        "\t",
    )
    std_sxz = ppsci.utils.reader.load_csv_file(
        "./data/shear_xz.txt",
        ("sigma_xz",),
        {"sigma_xz": "Shear Stress (Pa)"},
        "\t",
    )
    std_syz = ppsci.utils.reader.load_csv_file(
        "./data/shear_yz.txt",
        ("sigma_yz",),
        {"sigma_yz": "Shear Stress (Pa)"},
        "\t",
    )
    input_dict = {
        "x": std_xyzu["x"],
        "y": std_xyzu["y"],
        "z": std_xyzu["z"],
    }
    label_dict = {
        "u": std_xyzu["u"] / characteristic_displacement,
        "v": std_v["v"] / characteristic_displacement,
        "w": std_w["w"] / characteristic_displacement,
        "sigma_xx": std_sxx["sigma_xx"] * sigma_normalization,
        "sigma_yy": std_syy["sigma_yy"] * sigma_normalization,
        "sigma_zz": std_szz["sigma_zz"] * sigma_normalization,
        "sigma_xy": std_sxy["sigma_xy"] * sigma_normalization,
        "sigma_xz": std_sxz["sigma_xz"] * sigma_normalization,
        "sigma_yz": std_syz["sigma_yz"] * sigma_normalization,
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
            "weight": {key: np.ones_like(label_dict[key]) for key in label_dict},
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
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visulzie_u_v": ppsci.visualize.VisualizerVtu(
            input_dict,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
            prefix="result_u_v_w",
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
        iters_per_epoch,
        eval_during_train=True,
        log_freq=1,
        eval_freq=200,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    exit()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()

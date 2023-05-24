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
from paddle import fluid

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    fluid.core._set_prim_all_enabled(True)

    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./output_darcy2d" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("x", "y"), ("p",), 5, 20, "tanh", False, False)

    # set equation
    equation = {"Poisson": ppsci.equation.Poisson(2)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((0.0, 0.0), (1.0, 1.0))}

    # set dataloader config
    ITERS_PER_EPOCH = 1
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
    }

    NPOINT_PDE = 99**2
    NPOINT_TOP = 101
    NPOINT_BOTTOM = 101
    NPOINT_LEFT = 99
    NPOINT_RIGHT = 99

    # set constraint
    def poisson_ref_compute_func(_in):
        return (
            -8.0
            * (np.pi**2)
            * np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        )

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="EQ",
    )
    bc_top = ppsci.constraint.BoundaryConstraint(
        {"p": lambda out: out["p"]},
        {
            "p": lambda _in: np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        },
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_TOP},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(y, 1.0),
        name="BC_top",
    )
    bc_bottom = ppsci.constraint.BoundaryConstraint(
        {"p": lambda out: out["p"]},
        {
            "p": lambda _in: np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        },
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_BOTTOM},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(y, 0.0),
        name="BC_bottom",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"p": lambda out: out["p"]},
        {
            "p": lambda _in: np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        },
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_LEFT},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, 0.0),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"p": lambda out: out["p"]},
        {
            "p": lambda _in: np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        },
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_RIGHT},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, 1.0),
        name="BC_right",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_top.name: bc_top,
        bc_bottom.name: bc_bottom,
        bc_left.name: bc_left,
        bc_right.name: bc_right,
    }

    # set training hyper-parameters
    EPOCHS = 10000 if not args.epochs else args.epochs

    # set optimizer
    optimizer = ppsci.optimizer.Adam(1e-3)((model,))

    # set validator
    NPOINT_EVAL = NPOINT_PDE
    residual_validator = ppsci.validate.GeometryValidator(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": 8192,
            "sampler": {"name": "BatchSampler"},
        },
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # set visualizer(optional)
    # manually collate input data for visualization,
    NPOINT_BC = NPOINT_TOP + NPOINT_BOTTOM + NPOINT_LEFT + NPOINT_RIGHT
    vis_points = geom["rect"].sample_interior(NPOINT_PDE + NPOINT_BC, evenly=True)
    visualizer = {
        "visulzie_p": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"p": lambda d: d["p"]},
            prefix="result_p",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=200,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
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
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()

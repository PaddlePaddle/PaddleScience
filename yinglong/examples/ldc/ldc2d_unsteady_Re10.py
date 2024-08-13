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

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = "./ldc2d_unsteady_Re10" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(
        ("t", "x", "y"), ("u", "v", "p"), 9, 50, "tanh", False, False
    )
    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, True)}

    # set timestamps(including initial t0)
    timestamps = np.linspace(0.0, 1.5, 16, endpoint=True)
    # set time-geometry
    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(0.0, 1.5, timestamps=timestamps),
            ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05)),
        )
    }

    # set dataloader config
    ITERS_PER_EPOCH = 1
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
    }

    # pde/bc constraint use t1~tn, initial constraint use t0
    NTIME_ALL = len(timestamps)
    NPOINT_PDE, NTIME_PDE = 99**2, NTIME_ALL - 1
    NPOINT_TOP, NTIME_TOP = 101, NTIME_ALL - 1
    NPOINT_DOWN, NTIME_DOWN = 101, NTIME_ALL - 1
    NPOINT_LEFT, NTIME_LEFT = 99, NTIME_ALL - 1
    NPOINT_RIGHT, NTIME_RIGHT = 99, NTIME_ALL - 1
    NPOINT_IC, NTIME_IC = 99**2, 1

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_PDE * NTIME_PDE},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        weight_dict={
            "continuity": 0.0001,
            "momentum_x": 0.0001,
            "momentum_y": 0.0001,
        },
        name="EQ",
    )
    bc_top = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 1, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_TOP * NTIME_TOP},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(y, 0.05),
        name="BC_top",
    )
    bc_down = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_DOWN * NTIME_DOWN},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(y, -0.05),
        name="BC_down",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_LEFT * NTIME_LEFT},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(x, -0.05),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_RIGHT * NTIME_RIGHT},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(x, 0.05),
        name="BC_right",
    )
    ic = ppsci.constraint.InitialConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_IC * NTIME_IC},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="IC",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_top.name: bc_top,
        bc_down.name: bc_down,
        bc_left.name: bc_left,
        bc_right.name: bc_right,
        ic.name: ic,
    }

    # set training hyper-parameters
    EPOCHS = 20000 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        warmup_epoch=int(0.05 * EPOCHS),
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # set validator
    NPOINT_EVAL = NPOINT_PDE * NTIME_ALL
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"momentum_x": 0, "continuity": 0, "momentum_y": 0},
        geom["time_rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": 8192,
            "sampler": {"name": "BatchSampler"},
        },
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # set visualizer(optional)
    NPOINT_BC = NPOINT_TOP + NPOINT_DOWN + NPOINT_LEFT + NPOINT_RIGHT
    vis_initial_points = geom["time_rect"].sample_initial_interior(
        (NPOINT_IC + NPOINT_BC), evenly=True
    )
    vis_pde_points = geom["time_rect"].sample_interior(
        (NPOINT_PDE + NPOINT_BC) * NTIME_PDE, evenly=True
    )
    vis_points = vis_initial_points
    # manually collate input data for visualization,
    # (interior+boundary) x all timestamps
    for t in range(NTIME_PDE):
        for key in vis_points:
            vis_points[key] = np.concatenate(
                (
                    vis_points[key],
                    vis_pde_points[key][
                        t
                        * (NPOINT_PDE + NPOINT_BC) : (t + 1)
                        * (NPOINT_PDE + NPOINT_BC)
                    ],
                )
            )

    visualizer = {
        "visulzie_u_v": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            num_timestamps=NTIME_ALL,
            prefix="result_u_v",
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

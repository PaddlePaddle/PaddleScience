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
    output_dir = "./ldc2d_steady_Re10" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("x", "y"), ("u", "v", "p"), 9, 50, "tanh", False, False)

    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, False)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))}

    # set dataloader config
    iters_per_epoch = 1
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
    }

    npoint_pde = 99**2
    npoint_top = 101
    npoint_bottom = 101
    npoint_left = 99
    npoint_right = 99

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": npoint_pde},
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
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": npoint_top},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(y, 0.05),
        name="BC_top",
    )
    bc_bottom = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": npoint_bottom},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(y, -0.05),
        name="BC_bottom",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": npoint_left},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, -0.05),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": npoint_right},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, 0.05),
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
    epochs = 20000 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        epochs,
        iters_per_epoch,
        0.001,
        warmup_epoch=int(0.05 * epochs),
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # set validator
    npoints_eval = npoint_pde
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"momentum_x": 0, "continuity": 0, "momentum_y": 0},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": npoints_eval,
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
    npoint_bc = npoint_top + npoint_bottom + npoint_left + npoint_right
    vis_interior_points = geom["rect"].sample_interior(npoint_pde, evenly=True)
    vis_boundary_points = geom["rect"].sample_boundary(npoint_bc, evenly=True)

    # manually collate input data for visualization,
    # interior+boundary
    vis_points = {}
    for key in vis_interior_points:
        vis_points[key] = np.concatenate(
            (vis_interior_points[key], vis_boundary_points[key])
        )

    visualizer = {
        "visulzie_u_v": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            prefix="result_u_v",
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

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This code is based on PaddleScience/ppsci API
import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set training hyper-parameters
    EPOCHS = 20000 if not args.epochs else args.epochs
    ITERS_PER_EPOCH = 1
    EVAL_FREQ = 200

    # set output directory
    OUTPUT_DIR = "./output/laplace2d" if not args.output_dir else args.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("x", "y"), ("u",), 5, 20)

    # set equation
    equation = {"laplace": ppsci.equation.pde.Laplace(dim=2)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((0.0, 0.0), (1.0, 1.0))}

    # compute ground truth function
    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x, y = out["x"], out["y"]
        return np.cos(x) * np.cosh(y)

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
    }

    NPOINT_INTERIOR = 99**2
    NPOINT_BC = 400
    NPOINT_TOTAL = NPOINT_INTERIOR + NPOINT_BC

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["laplace"].equations,
        {"laplace": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_TOTAL},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="EQ",
    )
    bc = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_BC},
        ppsci.loss.MSELoss("sum"),
        name="BC",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc.name: bc,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=0.001)((model,))

    # set validator
    mse_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": NPOINT_TOTAL,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="MSE_Metric",
    )
    validator = {mse_metric.name: mse_metric}

    # set visualizer(optional)
    vis_points = geom["rect"].sample_interior(NPOINT_TOTAL, evenly=True)
    visualizer = {
        "visulzie_u": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"]},
            num_timestamps=1,
            prefix="result_u",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=EVAL_FREQ,
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

    # directly evaluate model from pretrained_model_path(optional)
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
    # visualize prediction from pretrained_model_path(optional)
    solver.visualize()

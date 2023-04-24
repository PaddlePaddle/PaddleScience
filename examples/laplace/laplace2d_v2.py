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
    epochs = 20000 if not args.epochs else args.epochs
    iters_per_epoch = 1
    eval_freq = 200

    # set output directory
    output_dir = "./output/laplace2d" if not args.output_dir else args.output_dir
    logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

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
        "iters_per_epoch": iters_per_epoch,
    }
    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["laplace"].equations,
        {"laplace": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": 9800},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="EQ",
    )
    bc = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": 400},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, 0.0)
        | np.isclose(x, 1.0)
        | np.isclose(y, 0.0)
        | np.isclose(y, 1.0),
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
    total_size = 9800
    mse_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": total_size,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="MSE_Metric",
    )
    validator = {mse_metric.name: mse_metric}

    # set visualizer(optional)
    vis_points = geom["rect"].sample_interior(total_size, evenly=True)
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
        output_dir,
        optimizer,
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        eval_during_train=True,
        eval_freq=eval_freq,
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
    logger.init_logger("ppsci", f"{output_dir}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        output_dir,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction from pretrained_model_path(optional)
    solver.visualize()

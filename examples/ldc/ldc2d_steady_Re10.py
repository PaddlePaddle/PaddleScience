"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

import ppsci

if __name__ == "__main__":
    ppsci.utils.misc.set_random_seed(42)
    model = ppsci.arch.MLP(["x", "y"], ["u", "v", "p"], 9, 50, "tanh", False, False)
    equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, False)}
    geom = {"rect": ppsci.geometry.Rectangle([-0.05, -0.05], [0.05, 0.05])}
    iters_per_epoch = 1
    constraint = {
        "EQ": ppsci.constraint.InteriorConstraint(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 9801,
                "num_workers": 2,
                "seed": 42,
                "use_shared_memory": False,
            },
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            weight_dict={
                "continuity": 0.0001,
                "momentum_x": 0.0001,
                "momentum_y": 0.0001,
            },
            name="EQ",
        ),
        "BC_top": ppsci.constraint.BoundaryConstraint(
            {"u": lambda out: out["u"], "v": lambda out: out["v"]},
            {"u": 1.0, "v": 0.0},
            geom["rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 101,
                "num_workers": 1,
                "seed": 42,
                "use_shared_memory": False,
            },
            ppsci.loss.MSELoss("sum"),
            criteria=lambda x, y: np.isclose(y, 0.05),
            weight_dict={"u": lambda input: 1.0 - 20.0 * input["x"]},
            name="BC_top",
        ),
        "BC_down": ppsci.constraint.BoundaryConstraint(
            {"u": lambda out: out["u"], "v": lambda out: out["v"]},
            {"u": 0.0, "v": 0.0},
            geom["rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 101,
                "num_workers": 1,
                "seed": 42,
                "use_shared_memory": False,
            },
            ppsci.loss.MSELoss("sum"),
            criteria=lambda x, y: np.isclose(y, -0.05),
            name="BC_down",
        ),
        "BC_left": ppsci.constraint.BoundaryConstraint(
            {"u": lambda out: out["u"], "v": lambda out: out["v"]},
            {"u": 0.0, "v": 0.0},
            geom["rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 99,
                "num_workers": 1,
                "seed": 42,
                "use_shared_memory": False,
            },
            ppsci.loss.MSELoss("sum"),
            criteria=lambda x, y: np.isclose(x, -0.05),
            name="BC_left",
        ),
        "BC_right": ppsci.constraint.BoundaryConstraint(
            {"u": lambda out: out["u"], "v": lambda out: out["v"]},
            {"u": 0.0, "v": 0.0},
            geom["rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 99,
                "num_workers": 1,
                "seed": 42,
                "use_shared_memory": False,
            },
            ppsci.loss.MSELoss("sum"),
            criteria=lambda x, y: np.isclose(x, 0.05),
            name="BC_right",
        ),
    }
    epochs = 20000
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        epochs,
        iters_per_epoch,
        0.001,
        warmup_epoch=int(0.05 * epochs),
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)([model])

    validator = {
        "Residual": ppsci.validate.GeometryValidator(
            equation["NavierStokes"].equations,
            {
                "momentum_x": 0,
                "continuity": 0,
                "momentum_y": 0,
                "u": 0.0,
                "v": 0.0,
                "p": 0.0,
            },
            geom["rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "total_size": 9801,
                "num_workers": 0,
                "seed": 42,
                "use_shared_memory": False,
            },
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            metric={"MSE": ppsci.metric.MSE()},
            name="Residual",
        )
    }

    output_dir = "./ldc2d_steady_Re10"
    train_solver = ppsci.solver.Solver(
        "train",
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
    )
    train_solver.train()

    # evaluate the final checkpoint
    eval_solver = ppsci.solver.Solver(
        "eval",
        model,
        constraint,
        output_dir,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model=f"./{output_dir}/checkpoints/epoch_{epochs}",
    )
    eval_solver.eval()

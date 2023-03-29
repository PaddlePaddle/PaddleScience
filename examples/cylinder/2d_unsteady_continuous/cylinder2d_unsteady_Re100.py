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
    model = ppsci.arch.MLP(
        ["t", "x", "y"], ["u", "v", "p"], 5, 50, "tanh", False, False
    )
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(0.02, 1.0, dim=2, time=True)
    }

    train_timestamp = np.linspace(1, 50, 50, endpoint=True).astype("float32")
    train_timestamp = np.random.choice(train_timestamp, 30)
    train_timestamp.sort()

    val_timestamp = np.linspace(1, 50, 50, endpoint=True).astype("float32")

    print(f"train_timestamp: {train_timestamp}")
    print(f"val_timestamp: {val_timestamp}")

    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(
                1,
                50,
                timestamps=np.concatenate(
                    (np.array([1], dtype="float32"), train_timestamp), axis=0
                ),
            ),
            ppsci.geometry.PointCloud(
                "./datasets/domain_train.csv",
                ["Points:0", "Points:1"],
                alias_dict={"Points:0": "x", "Points:1": "y"},
            ),
        ),
        "time_rect_eval": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(1, 50, timestamps=val_timestamp),
            ppsci.geometry.PointCloud(
                "./datasets/domain_train.csv",
                ["Points:0", "Points:1"],
                alias_dict={"Points:0": "x", "Points:1": "y"},
            ),
        ),
    }
    iters_per_epoch = 1
    constraint = {
        "EQ": ppsci.constraint.InteriorConstraint(
            equation["NavierStokes"].equations,
            {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
            geom["time_rect"],
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 282600,
            },
            ppsci.loss.MSELoss("mean"),
            name="EQ",
        ),
        "BC_inlet": ppsci.constraint.SupervisedConstraint(
            "./datasets/domain_inlet_cylinder.csv",
            ["Points:0", "Points:1"],
            ["U:0", "U:1"],
            {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v"},
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 4830,
            },
            ppsci.loss.MSELoss("mean"),
            {"u": 10.0, "v": 10.0},
            timestamps=train_timestamp,
            name="BC_inlet",
        ),
        "BC_outlet": ppsci.constraint.SupervisedConstraint(
            "./datasets/domain_outlet.csv",
            ["Points:0", "Points:1"],
            ["p"],
            {"Points:0": "x", "Points:1": "y"},
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 2430,
            },
            ppsci.loss.MSELoss("mean"),
            timestamps=train_timestamp,
            name="BC_outlet",
        ),
        "IC": ppsci.constraint.SupervisedInitialConstraint(
            "./datasets/initial/ic0.1.csv",
            ["Points:0", "Points:1"],
            ["U:0", "U:1", "p"],
            1,
            {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v"},
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 9420,
            },
            ppsci.loss.MSELoss("mean"),
            {"u": 10.0, "v": 10.0, "p": 10.0},
            name="IC",
        ),
        "Sup": ppsci.constraint.SupervisedConstraint(
            "./datasets/probe/probe1_50.csv",
            ["t", "Points:0", "Points:1"],
            ["U:0", "U:1"],
            {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v"},
            {
                "dataset": "IterableNamedArrayDataset",
                "iters_per_epoch": iters_per_epoch,
                "batch_size": 8490,
            },
            ppsci.loss.MSELoss("mean"),
            {"u": 10.0, "v": 10.0},
            timestamps=train_timestamp,
            name="Sup",
        ),
    }
    epochs = 40000
    lr_scheduler = None
    optimizer = ppsci.optimizer.Adam(0.001)([model])

    validator = {
        "Residual": ppsci.validate.CSVValidator(
            "./datasets/cylinder2d_eval_points.csv",
            ["t", "x", "y"],
            ["u"],
            {},
            {
                "dataset": "IterableNamedArrayDataset",
                "total_size": 471000,
            },
            ppsci.loss.MSELoss("mean"),
            metric={"MSE": ppsci.metric.MSE()},
            name="Residual",
        ),
    }
    output_dir = "./output_cylinder2d_unsteady"

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

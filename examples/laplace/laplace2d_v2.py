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

if __name__ == "__main__":

    output_dir = "./output/laplace2d"
    epochs = 20000
    iters_per_epoch = 1

    ppsci.utils.misc.set_random_seed(42)

    # manually init model
    model = ppsci.arch.MLP(["x", "y"], ["u"], 5, 20, "tanh")

    # manually init geometry(ies)
    geom = {"rect": ppsci.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])}

    # manually init equation(s)
    equation = {"laplace": ppsci.equation.pde.Laplace(dim=2)}

    # maunally build constraint(s)
    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x, y = out["x"], out["y"]
        return np.cos(x) * np.cosh(y)

    eq_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "batch_size": 9800,
        "iters_per_epoch": iters_per_epoch,
    }

    bc_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "batch_size": 400,
        "iters_per_epoch": iters_per_epoch,
    }
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["laplace"].equations,
        {"laplace": lambda out: 0.0},
        geom["rect"],
        eq_dataloader_cfg,
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="EQ",
    )
    bc = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        bc_dataloader_cfg,
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, 0.0)
        | np.isclose(x, 1.0)
        | np.isclose(y, 0.0)
        | np.isclose(y, 1.0),
        name="BC",
    )
    constraint = {
        pde_constraint.name: pde_constraint,
        bc.name: bc,
    }

    # init optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=0.001)([model])

    # maunally build validator
    eval_dataloader = {
        "dataset": "IterableNamedArrayDataset",
        "total_size": 9800,
    }
    mse_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["rect"],
        eval_dataloader,
        ppsci.loss.MSELoss("mean"),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="MSE_Metric",
    )
    validator = {mse_metric.name: mse_metric}

    train_solver = ppsci.solver.Solver(
        "train",
        model,
        constraint,
        output_dir,
        optimizer,
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        eval_during_train=True,
        eval_freq=200,
        equation=equation,
        geom=geom,
        validator=validator,
    )
    train_solver.train()

    eval_solver = ppsci.solver.Solver(
        "eval",
        model,
        constraint,
        output_dir,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    eval_solver.eval()

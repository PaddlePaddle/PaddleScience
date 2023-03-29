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
from ppsci import autodiff

if __name__ == "__main__":

    output_dir = "output/laplace2d"
    epochs = 20000
    iters_per_epoch = 1

    ppsci.utils.misc.set_random_seed(42)

    # manually init model
    model = ppsci.arch.MLP(["x", "y"], ["u"], 5, 20, "tanh")

    # manually init geometry(ies)
    geom = {"rect": ppsci.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])}

    # manually init equation(s)
    def laplace_compute_func(d):
        x, y = d["x"], d["y"]
        u = d["u"]
        laplace = autodiff.hessian(u, x) + autodiff.hessian(u, y)
        return laplace

    laplace_equation = ppsci.equation.pde.PDE()
    laplace_equation.add_equation("laplace", laplace_compute_func)
    equation = {"laplace": laplace_equation}

    # maunally build constraint(s)
    def u_compute_func(d):
        x, y = d["x"], d["y"]
        return np.cos(x) * np.cosh(y)

    eq_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "batch_size": 9800,
        "iters_per_epoch": iters_per_epoch,
    }

    # init constraint(s)
    bc_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "batch_size": 400,
        "iters_per_epoch": iters_per_epoch,
    }

    # maunally build constraint(s)
    constraint = {
        "EQ": ppsci.constraint.InteriorConstraint(
            equation["laplace"].equations,
            {"laplace": lambda d: 0.0},
            geom["rect"],
            eq_dataloader_cfg,
            ppsci.loss.MSELoss("sum"),
            evenly=True,
            name="EQ",
        ),
        "BC": ppsci.constraint.BoundaryConstraint(
            {"u": lambda d: d["u"]},
            {"u": u_compute_func},
            geom["rect"],
            bc_dataloader_cfg,
            ppsci.loss.MSELoss("sum"),
            criteria=lambda x, y: np.isclose(x, 0.0)
            | np.isclose(x, 1.0)
            | np.isclose(y, 0.0)
            | np.isclose(y, 1),
            name="BC",
        ),
    }

    # init optimizer and lr scheduler
    lr_scheduler = ppsci.optimizer.lr_scheduler.ConstLR(
        epochs, iters_per_epoch, 0.001, by_epoch=False
    )()
    optimizer = ppsci.optimizer.Adam(
        lr_scheduler,
    )([model])

    # maunally build validator
    dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "total_size": 9800,
    }

    validator = {
        "Residual": ppsci.validate.GeometryValidator(
            {"u": lambda d: d["u"]},
            {"u": u_compute_func},
            geom["rect"],
            dataloader_cfg,
            ppsci.loss.MSELoss("mean"),
            evenly=True,
            metric={"MSE": ppsci.metric.MSE()},
            with_initial=True,
            name="Residual",
        )
    }

    train_solver = ppsci.solver.Solver(
        "train",
        model,
        constraint,
        output_dir,
        optimizer,
        lr_scheduler,
        epochs=epochs,
        iters_per_epoch=iters_per_epoch,
        eval_during_train=True,
        eval_freq=200,
        equation=equation,
        geom=geom,
        validator=validator,
    )
    train_solver.train()

    # eval_solver = ppsci.solver.Solver(
    #         "eval",
    #         model,
    #         constraint,
    #         output_dir,
    #         equation=equation,
    #         geom=geom,
    #         validator=validator,
    #         pretrained_model_path=f"./{output_dir}/checkpoints/best_model",
    #     )
    # eval_solver.eval()

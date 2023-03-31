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
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    output_dir = "./ldc2d_unsteady_Re10"

    # set model
    model = ppsci.arch.MLP(
        ["t", "x", "y"], ["u", "v", "p"], 9, 50, "tanh", False, False
    )
    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(0.01, 1.0, 2, True)}

    # timestamps including initial t0
    timestamps = np.linspace(0.0, 1.5, 16, endpoint=True)
    # set time-eometry
    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(0.0, 1.5, timestamps=timestamps),
            ppsci.geometry.Rectangle([-0.05, -0.05], [0.05, 0.05]),
        )
    }

    # set dataloader config
    iters_per_epoch = 1
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
    }

    # pde/bc constraint use t1~tn, initial constraint use t0
    ntime_all = len(timestamps)
    npoint_pde, ntime_pde = 9801, ntime_all - 1
    npoint_top, ntime_top = 101, ntime_all - 1
    npoint_down, ntime_down = 101, ntime_all - 1
    npoint_left, ntime_left = 99, ntime_all - 1
    npoint_right, ntime_right = 99, ntime_all - 1
    npoint_ic, ntime_ic = 9801, 1

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, **{"batch_size": npoint_pde * ntime_pde}},
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
        {**train_dataloader_cfg, **{"batch_size": npoint_top * ntime_top}},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(y, 0.05),
        weight_dict={"u": lambda input: 1 - 20 * input["x"]},
        name="BC_top",
    )
    bc_down = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, **{"batch_size": npoint_down * ntime_down}},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(y, -0.05),
        name="BC_down",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, **{"batch_size": npoint_left * ntime_left}},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(x, -0.05),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, **{"batch_size": npoint_right * ntime_right}},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda t, x, y: np.isclose(x, 0.05),
        name="BC_right",
    )
    ic = ppsci.constraint.InitialConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["time_rect"],
        {**train_dataloader_cfg, **{"batch_size": npoint_ic * ntime_ic}},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="IC",
    )
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_top.name: bc_top,
        bc_down.name: bc_down,
        bc_left.name: bc_left,
        bc_right.name: bc_right,
        ic.name: ic,
    }

    # set training hyper-parameters
    epochs = 20000
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        epochs,
        iters_per_epoch,
        0.001,
        warmup_epoch=int(0.05 * epochs),
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)([model])

    # set validator
    npoints_eval = npoint_pde * ntime_all
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"momentum_x": 0, "continuity": 0, "momentum_y": 0},
        geom["time_rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": npoints_eval,
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
    npoint_bc = npoint_top + npoint_down + npoint_left + npoint_right
    ntime_bc = ntime_top
    vis_initial_points = geom["time_rect"].sample_initial_interior(
        npoint_ic, evenly=True
    )
    vis_interior_points = geom["time_rect"].sample_interior(
        npoint_pde * ntime_pde, evenly=True
    )
    vis_boundary_points = geom["time_rect"].sample_boundary(
        npoint_bc * ntime_bc, evenly=True
    )

    # concatenate interior points with boundary points
    vis_initial_points = {
        key: np.concatenate(
            (vis_initial_points[key], vis_boundary_points[key][:npoint_bc])
        )
        for key in vis_initial_points
    }

    vis_points = vis_initial_points
    for t in range(ntime_pde):
        for key in vis_interior_points:
            vis_points[key] = np.concatenate(
                (
                    vis_points[key],
                    vis_interior_points[key][t * npoint_pde : (t + 1) * npoint_pde],
                    vis_boundary_points[key][t * npoint_bc : (t + 1) * npoint_bc],
                )
            )

    visualizer = {
        "visulzie_u_v": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            ntime_all,
            "result_u_v",
        )
    }

    # initialize train solver
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
        visualizer=visualizer,
    )
    # train model
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
        visualizer=visualizer,
        pretrained_model_path=f"{output_dir}/checkpoints/latest",
    )
    eval_solver.eval()

    eval_solver.visualize()

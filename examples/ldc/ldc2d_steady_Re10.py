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

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(cfg.NU, cfg.RHO, 2, False)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))}

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    NPOINT_PDE = 99**2
    NPOINT_TOP = 101
    NPOINT_BOTTOM = 101
    NPOINT_LEFT = 99
    NPOINT_RIGHT = 99

    # set constraint
    pde = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        weight_dict=cfg.TRAIN.weight.pde,
        name="EQ",
    )
    bc_top = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 1, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_TOP},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(y, 0.05),
        name="BC_top",
    )
    bc_bottom = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_BOTTOM},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(y, -0.05),
        name="BC_bottom",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_LEFT},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, -0.05),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_RIGHT},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y: np.isclose(x, 0.05),
        name="BC_right",
    )
    # wrap constraints together
    constraint = {
        pde.name: pde,
        bc_top.name: bc_top,
        bc_bottom.name: bc_bottom,
        bc_left.name: bc_left,
        bc_right.name: bc_right,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(
        **cfg.TRAIN.lr_scheduler,
        warmup_epoch=int(0.05 * cfg.TRAIN.epochs),
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    NPOINT_EVAL = NPOINT_PDE
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"momentum_x": 0, "continuity": 0, "momentum_y": 0},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": cfg.EVAL.batch_size.residual_validator,
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
        "visualize_u_v": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            prefix="result_u_v",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(cfg.NU, cfg.RHO, 2, False)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-0.05, -0.05), (0.05, 0.05))}

    NPOINT_PDE = 99**2
    NPOINT_TOP = 101
    NPOINT_BOTTOM = 101
    NPOINT_LEFT = 99
    NPOINT_RIGHT = 99

    # set validator
    NPOINT_EVAL = NPOINT_PDE
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"momentum_x": 0, "continuity": 0, "momentum_y": 0},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": cfg.EVAL.batch_size.residual_validator,
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
        "visualize_u_v": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            prefix="result_u_v",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()


@hydra.main(
    version_base=None, config_path="./conf", config_name="ldc2d_steady_Re10.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

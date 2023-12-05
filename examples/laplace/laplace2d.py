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

from os import path as osp

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"laplace": ppsci.equation.Laplace(dim=2)}

    # set geometry
    geom = {
        "rect": ppsci.geometry.Rectangle(
            cfg.DIAGONAL_COORD.xmin, cfg.DIAGONAL_COORD.xmax
        )
    }

    # compute ground truth function
    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x, y = out["x"], out["y"]
        return np.cos(x) * np.cosh(y)

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    NPOINT_TOTAL = cfg.NPOINT_INTERIOR + cfg.NPOINT_BC

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
        {**train_dataloader_cfg, "batch_size": cfg.NPOINT_BC},
        ppsci.loss.MSELoss("sum"),
        name="BC",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc.name: bc,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=cfg.TRAIN.learning_rate)(model)

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
        "visualize_u": ppsci.visualize.VisualizerVtu(
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
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
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


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"laplace": ppsci.equation.Laplace(dim=2)}

    # set geometry
    geom = {
        "rect": ppsci.geometry.Rectangle(
            cfg.DIAGONAL_COORD.xmin, cfg.DIAGONAL_COORD.xmax
        )
    }

    # compute ground truth function
    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x, y = out["x"], out["y"]
        return np.cos(x) * np.cosh(y)

    NPOINT_TOTAL = cfg.NPOINT_INTERIOR + cfg.NPOINT_BC

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
        "visualize_u": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"]},
            num_timestamps=1,
            prefix="result_u",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()
    # visualize prediction
    solver.visualize()


@hydra.main(version_base=None, config_path="./conf", config_name="laplace2d.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

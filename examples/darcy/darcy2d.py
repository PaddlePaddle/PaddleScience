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

from os import path as osp

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"Poisson": ppsci.equation.Poisson(2)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((0.0, 0.0), (1.0, 1.0))}

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    NPOINT_PDE = cfg.NPOINT_PDE
    NPOINT_BC = cfg.NPOINT_BC

    # set constraint
    def poisson_ref_compute_func(_in):
        return (
            -8.0
            * (np.pi**2)
            * np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        )

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="EQ",
    )

    bc = ppsci.constraint.BoundaryConstraint(
        {"p": lambda out: out["p"]},
        {
            "p": lambda _in: np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        },
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
    lr_scheduler = ppsci.optimizer.lr_scheduler.OneCycleLR(**cfg.TRAIN.lr_scheduler)()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    NPOINT_EVAL = NPOINT_PDE
    residual_validator = ppsci.validate.GeometryValidator(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": cfg.TRAIN.batch_size.residual_validator,
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
    vis_points = geom["rect"].sample_interior(NPOINT_PDE + NPOINT_BC, evenly=True)
    visualizer = {
        "visulzie_p": ppsci.visualize.VisualizerVtu(
            vis_points,
            {
                "p": lambda d: d["p"],
                "p_ref": lambda d: paddle.sin(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"]),
                "p_diff": lambda d: paddle.sin(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"])
                - d["p"],
                "ux": lambda d: jacobian(d["p"], d["x"]),
                "ux_ref": lambda d: 2
                * np.pi
                * paddle.cos(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"]),
                "ux_diff": lambda d: jacobian(d["p"], d["x"])
                - 2
                * np.pi
                * paddle.cos(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"]),
                "uy": lambda d: jacobian(d["p"], d["y"]),
                "uy_ref": lambda d: -2
                * np.pi
                * paddle.sin(2 * np.pi * d["x"])
                * paddle.sin(2 * np.pi * d["y"]),
                "uy_diff": lambda d: jacobian(d["p"], d["y"])
                - (
                    -2
                    * np.pi
                    * paddle.sin(2 * np.pi * d["x"])
                    * paddle.sin(2 * np.pi * d["y"])
                ),
            },
            prefix="result_p_ux_uy",
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
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # finetuning pretrained model with L-BFGS
    OUTPUT_DIR = cfg.TRAIN.lbfgs.output_dir
    logger.init_logger("ppsci", osp.join(OUTPUT_DIR, f"{cfg.mode}.log"), "info")
    EPOCHS = cfg.TRAIN.epochs // 10
    optimizer_lbfgs = ppsci.optimizer.LBFGS(
        cfg.TRAIN.lbfgs.learning_rate, cfg.TRAIN.lbfgs.max_iter
    )(model)
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer_lbfgs,
        None,
        EPOCHS,
        cfg.TRAIN.lbfgs.iters_per_epoch,
        eval_during_train=cfg.TRAIN.lbfgs.eval_during_train,
        eval_freq=cfg.TRAIN.lbfgs.eval_freq,
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
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"Poisson": ppsci.equation.Poisson(2)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((0.0, 0.0), (1.0, 1.0))}

    NPOINT_PDE = cfg.NPOINT_PDE
    NPOINT_BC = cfg.NPOINT_BC

    # set constraint
    def poisson_ref_compute_func(_in):
        return (
            -8.0
            * (np.pi**2)
            * np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        )

    # set validator
    NPOINT_EVAL = NPOINT_PDE
    residual_validator = ppsci.validate.GeometryValidator(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": cfg.TRAIN.batch_size.residual_validator,
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
    vis_points = geom["rect"].sample_interior(NPOINT_PDE + NPOINT_BC, evenly=True)
    visualizer = {
        "visulzie_p": ppsci.visualize.VisualizerVtu(
            vis_points,
            {
                "p": lambda d: d["p"],
                "p_ref": lambda d: paddle.sin(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"]),
                "p_diff": lambda d: paddle.sin(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"])
                - d["p"],
                "ux": lambda d: jacobian(d["p"], d["x"]),
                "ux_ref": lambda d: 2
                * np.pi
                * paddle.cos(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"]),
                "ux_diff": lambda d: jacobian(d["p"], d["x"])
                - 2
                * np.pi
                * paddle.cos(2 * np.pi * d["x"])
                * paddle.cos(2 * np.pi * d["y"]),
                "uy": lambda d: jacobian(d["p"], d["y"]),
                "uy_ref": lambda d: -2
                * np.pi
                * paddle.sin(2 * np.pi * d["x"])
                * paddle.sin(2 * np.pi * d["y"]),
                "uy_diff": lambda d: jacobian(d["p"], d["y"])
                - (
                    -2
                    * np.pi
                    * paddle.sin(2 * np.pi * d["x"])
                    * paddle.sin(2 * np.pi * d["y"])
                ),
            },
            prefix="result_p_ux_uy",
        )
    }

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


@hydra.main(version_base=None, config_path="./conf", config_name="darcy2d.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

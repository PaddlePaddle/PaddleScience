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
        {**train_dataloader_cfg, "batch_size": cfg.NPOINT_PDE},
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
    lr_scheduler = ppsci.optimizer.lr_scheduler.OneCycleLR(**cfg.TRAIN.lr_scheduler)()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    residual_validator = ppsci.validate.GeometryValidator(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": cfg.NPOINT_PDE,
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
    vis_points = geom["rect"].sample_interior(
        cfg.NPOINT_PDE + cfg.NPOINT_BC, evenly=True
    )
    visualizer = {
        "visualize_p_ux_uy": ppsci.visualize.VisualizerVtu(
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

    # fine-tuning pretrained model with L-BFGS
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

    # set constraint
    def poisson_ref_compute_func(_in):
        return (
            -8.0
            * (np.pi**2)
            * np.sin(2.0 * np.pi * _in["x"])
            * np.cos(2.0 * np.pi * _in["y"])
        )

    # set validator
    residual_validator = ppsci.validate.GeometryValidator(
        equation["Poisson"].equations,
        {"poisson": poisson_ref_compute_func},
        geom["rect"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": cfg.NPOINT_PDE,
            "batch_size": cfg.EVAL.batch_size.residual_validator,
            "sampler": {"name": "BatchSampler"},
        },
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # set visualizer
    # manually collate input data for visualization,
    vis_points = geom["rect"].sample_interior(
        cfg.NPOINT_PDE + cfg.NPOINT_BC, evenly=True
    )
    visualizer = {
        "visualize_p_ux_uy": ppsci.visualize.VisualizerVtu(
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
    # visualize prediction
    solver.visualize()


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]

    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((0.0, 0.0), (1.0, 1.0))}
    # manually collate input data for visualization,
    input_dict = geom["rect"].sample_interior(
        cfg.NPOINT_PDE + cfg.NPOINT_BC, evenly=True
    )
    output_dict = predictor.predict(
        {key: input_dict[key] for key in cfg.MODEL.input_keys}, cfg.INFER.batch_size
    )
    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }
    ppsci.visualize.save_vtu_from_dict(
        "./visual/darcy2d.vtu",
        {**input_dict, **output_dict},
        input_dict.keys(),
        cfg.MODEL.output_keys,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="darcy2d.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

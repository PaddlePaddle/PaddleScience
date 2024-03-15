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
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger


def train(cfg: DictConfig):
    # enable computation for fourth-order differentiation of matmul
    paddle.framework.core.set_prim_eager_enabled(True)
    paddle.framework.core._set_prim_all_enabled(True)
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry
    geom = {"interval": ppsci.geometry.Interval(0, 1)}

    # set equation(s)
    equation = {"biharmonic": ppsci.equation.Biharmonic(dim=1, q=cfg.q, D=cfg.D)}

    # set dataloader config
    dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }
    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["biharmonic"].equations,
        {"biharmonic": 0},
        geom["interval"],
        {**dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.pde},
        ppsci.loss.MSELoss(),
        random="Hammersley",
        name="EQ",
    )
    bc = ppsci.constraint.BoundaryConstraint(
        {
            "u0": lambda d: d["u"][0:1],
            "u__x": lambda d: jacobian(d["u"], d["x"])[1:2],
            "u__x__x": lambda d: hessian(d["u"], d["x"])[2:3],
            "u__x__x__x": lambda d: jacobian(hessian(d["u"], d["x"]), d["x"])[3:4],
        },
        {"u0": 0, "u__x": 0, "u__x__x": 0, "u__x__x__x": 0},
        geom["interval"],
        {**dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc},
        ppsci.loss.MSELoss("sum"),
        evenly=True,
        name="BC",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc.name: bc,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x = out["x"]
        return -(x**4) / 24 + x**3 / 6 - x**2 / 4

    l2_rel_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["interval"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": cfg.EVAL.total_size,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Metric",
    )
    validator = {l2_rel_metric.name: l2_rel_metric}

    # set visualizer(optional)
    visu_points = geom["interval"].sample_interior(cfg.EVAL.total_size, evenly=True)
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerScatter1D(
            visu_points,
            ("x",),
            {
                "u_label": lambda d: u_solution_func(d),
                "u_pred": lambda d: d["u"],
            },
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
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        to_static=cfg.to_static,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def evaluate(cfg: DictConfig):
    # enable computation for fourth-order differentiation of matmul
    paddle.framework.core.set_prim_eager_enabled(True)
    paddle.framework.core._set_prim_all_enabled(True)
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry
    geom = {"interval": ppsci.geometry.Interval(0, 1)}

    # set equation(s)
    equation = {"biharmonic": ppsci.equation.Biharmonic(dim=1, q=cfg.q, D=cfg.D)}

    # set validator
    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x = out["x"]
        return -(x**4) / 24 + x**3 / 6 - x**2 / 4

    l2_rel_metric = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["interval"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": cfg.EVAL.total_size,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Metric",
    )
    validator = {l2_rel_metric.name: l2_rel_metric}

    # set visualizer(optional)
    visu_points = geom["interval"].sample_interior(cfg.EVAL.total_size, evenly=True)
    visualizer = {
        "visualize_u": ppsci.visualize.VisualizerScatter1D(
            visu_points,
            ("x",),
            {
                "u_label": lambda d: u_solution_func(d),
                "u_pred": lambda d: d["u"],
            },
            num_timestamps=1,
            prefix="result_u",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        None,
        cfg.output_dir,
        None,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        to_static=cfg.to_static,
    )
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
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
    geom = {"interval": ppsci.geometry.Interval(0, 1)}
    input_dict = geom["interval"].sample_interior(cfg.INFER.total_size, evenly=True)

    output_dict = predictor.predict({"x": input_dict["x"]}, cfg.INFER.batch_size)

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    def u_solution_func(out):
        """compute ground truth for u as label data"""
        x = out["x"]
        return -(x**4) / 24 + x**3 / 6 - x**2 / 4

    ppsci.visualize.save_plot_from_1d_dict(
        "./euler_beam_pred",
        {**input_dict, **output_dict, "u_label": u_solution_func(input_dict)},
        ("x",),
        ("u", "u_label"),
    )


@hydra.main(version_base=None, config_path="./conf", config_name="euler_beam.yaml")
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

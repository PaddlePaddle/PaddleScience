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

# Reference: https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py

from os import path as osp
from typing import Dict
from typing import Tuple

import hydra
import numpy as np
import paddle
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import jacobian


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry
    geom = {"timedomain": ppsci.geometry.TimeDomain(*cfg.BOUNDS)}

    # set equation
    def kernel_func(x, s):
        return np.exp(s - x)

    def func(out):
        x, u = out["x"], out["u"]
        return jacobian(u, x) + u

    equation = {
        "volterra": ppsci.equation.Volterra(
            cfg.BOUNDS[0],
            cfg.TRAIN.npoint_interior,
            cfg.TRAIN.quad_deg,
            kernel_func,
            func,
        )
    }

    # set constraint
    # set transform for input data
    def input_data_quad_transform(
        input: Dict[str, np.ndarray],
        weight: Dict[str, np.ndarray],
        label: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, paddle.Tensor], Dict[str, paddle.Tensor], Dict[str, paddle.Tensor]
    ]:
        """Get sampling points for integral.

        Args:
            input (Dict[str, paddle.Tensor]): Raw input dict.
            weight (Dict[str, paddle.Tensor]): Raw weight dict.
            label (Dict[str, paddle.Tensor]): Raw label dict.

        Returns:
            Tuple[ Dict[str, paddle.Tensor], Dict[str, paddle.Tensor], Dict[str, paddle.Tensor] ]:
                Input dict contained sampling points, weight dict and label dict.
        """
        x = input["x"]  # N points.
        x_quad = equation["volterra"].get_quad_points(x).reshape([-1, 1])  # NxQ
        x_quad = paddle.concat((x, x_quad), axis=0)  # M+MxQ: [M|Q1|Q2,...,QM|]
        return (
            {
                **input,
                "x": x_quad,
            },
            weight,
            label,
        )

    # interior constraint
    ide_constraint = ppsci.constraint.InteriorConstraint(
        equation["volterra"].equations,
        {"volterra": 0},
        geom["timedomain"],
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "transforms": (
                    {
                        "FunctionalTransform": {
                            "transform_func": input_data_quad_transform,
                        },
                    },
                ),
            },
            "batch_size": cfg.TRAIN.npoint_interior,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        evenly=True,
        name="EQ",
    )

    # initial condition
    def u_solution_func(in_):
        if isinstance(in_["x"], paddle.Tensor):
            return paddle.exp(-in_["x"]) * paddle.cosh(in_["x"])
        return np.exp(-in_["x"]) * np.cosh(in_["x"])

    ic = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["timedomain"],
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": cfg.TRAIN.npoint_ic,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        criteria=geom["timedomain"].on_initial,
        name="IC",
    )
    # wrap constraints together
    constraint = {
        ide_constraint.name: ide_constraint,
        ic.name: ic,
    }

    # set optimizer
    optimizer = ppsci.optimizer.LBFGS(**cfg.TRAIN.optimizer)(model)

    # set validator
    l2rel_validator = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["timedomain"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": cfg.EVAL.npoint_eval,
        },
        ppsci.loss.L2RelLoss(),
        evenly=True,
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Validator",
    )
    validator = {l2rel_validator.name: l2rel_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        equation=equation,
        validator=validator,
        cfg=cfg,
    )
    # train model
    solver.train()

    # visualize prediction after finished training
    input_data = geom["timedomain"].uniform_points(100)
    label_data = u_solution_func({"x": input_data})
    output_data = solver.predict({"x": input_data}, return_numpy=True)["u"]

    plt.plot(input_data, label_data, "-", label=r"$u(t)$")
    plt.plot(input_data, output_data, "o", label=r"$\hat{u}(t)$", markersize=4.0)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.title(r"$u-t$")
    plt.savefig(osp.join(cfg.output_dir, "./Volterra_IDE.png"), dpi=200)


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry
    geom = {"timedomain": ppsci.geometry.TimeDomain(*cfg.BOUNDS)}
    # set validator

    def u_solution_func(in_) -> np.ndarray:
        if isinstance(in_["x"], paddle.Tensor):
            return paddle.exp(-in_["x"]) * paddle.cosh(in_["x"])
        return np.exp(-in_["x"]) * np.cosh(in_["x"])

    l2rel_validator = ppsci.validate.GeometryValidator(
        {"u": lambda out: out["u"]},
        {"u": u_solution_func},
        geom["timedomain"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": cfg.EVAL.npoint_eval,
        },
        ppsci.loss.L2RelLoss(),
        evenly=True,
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Validator",
    )
    validator = {l2rel_validator.name: l2rel_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    # evaluate model
    solver.eval()

    # visualize prediction
    input_data = geom["timedomain"].uniform_points(cfg.EVAL.npoint_eval)
    label_data = u_solution_func({"x": input_data})
    output_data = solver.predict({"x": input_data}, return_numpy=True)["u"]

    plt.plot(input_data, label_data, "-", label=r"$u(t)$")
    plt.plot(input_data, output_data, "o", label=r"$\hat{u}(t)$", markersize=4.0)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.title(r"$u-t$")
    plt.savefig(osp.join(cfg.output_dir, "./Volterra_IDE.png"), dpi=200)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None, 1], "float32", name=key)
            for key in cfg.MODEL.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # set geometry
    geom = {"timedomain": ppsci.geometry.TimeDomain(*cfg.BOUNDS)}

    input_data = geom["timedomain"].uniform_points(cfg.EVAL.npoint_eval)
    input_dict = {"x": input_data}

    output_dict = predictor.predict(
        {key: input_dict[key] for key in cfg.MODEL.input_keys}, cfg.INFER.batch_size
    )

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    def u_solution_func(in_) -> np.ndarray:
        if isinstance(in_["x"], paddle.Tensor):
            return paddle.exp(-in_["x"]) * paddle.cosh(in_["x"])
        return np.exp(-in_["x"]) * np.cosh(in_["x"])

    label_data = u_solution_func({"x": input_data})
    output_data = output_dict["u"]

    # save result
    plt.plot(input_data, label_data, "-", label=r"$u(t)$")
    plt.plot(input_data, output_data, "o", label=r"$\hat{u}(t)$", markersize=4.0)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.title(r"$u-t$")
    plt.savefig("./Volterra_IDE_pred.png", dpi=200)


@hydra.main(version_base=None, config_path="./conf", config_name="volterra_ide.yaml")
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

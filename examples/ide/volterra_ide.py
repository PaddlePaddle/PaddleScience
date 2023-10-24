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
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.SEED)

    # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(("x",), ("u",), 3, 20)

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
    # set input transform
    def quad_transform(
        input: Dict[str, np.ndarray],
        weight: Dict[str, np.ndarray],
        label: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, paddle.Tensor], Dict[str, paddle.Tensor], Dict[str, paddle.Tensor]
    ]:
        """Get sampling points for integral.

        Args:
            input (Dict[str, paddle.Tensor]): Raw input dict.

        Returns:
            Dict[str, paddle.Tensor]: Input dict contained sampling points.
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
                            "transform_func": quad_transform,
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
    optimizer = ppsci.optimizer.LBFGS(**cfg.optimizer)(model)

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
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
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
    plt.savefig("./Volterra_IDE.png", dpi=200)


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.SEED)

    # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.MLP(("x",), ("u",), 3, 20)

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
        output_dir=cfg.output_dir,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # visualize prediction
    input_data = geom["timedomain"].uniform_points(100)
    label_data = u_solution_func({"x": input_data})
    output_data = solver.predict({"x": input_data}, return_numpy=True)["u"]

    plt.plot(input_data, label_data, "-", label=r"$u(t)$")
    plt.plot(input_data, output_data, "o", label=r"$\hat{u}(t)$", markersize=4.0)
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.title(r"$u-t$")
    plt.savefig("./Volterra_IDE.png", dpi=200)


@hydra.main(version_base=None, config_path="./conf", config_name="volterra_ide.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

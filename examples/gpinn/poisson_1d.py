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

# Reference: https://github.com/lu-group/gpinn/blob/main/src/poisson_1d.py

from os import path as osp
from typing import Dict

import hydra
import numpy as np
import paddle
import sympy as sp
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import logger


class gPINN1D(ppsci.equation.PDE):
    def __init__(self, invar: str, outvar: str):
        super().__init__()
        x = self.create_symbols(invar)
        u = self.create_function(outvar, (x,))

        dy_xx = u.diff(x, 2)
        dy_xxx = u.diff(x, 3)

        f = 8 * sp.sin(8 * x)
        for i in range(1, 5):
            f += i * sp.sin(i * x)

        df_x = (
            sp.cos(x)
            + 4 * sp.cos(2 * x)
            + 9 * sp.cos(3 * x)
            + 16 * sp.cos(4 * x)
            + 64 * sp.cos(8 * x)
        )

        self.add_equation("res1", -dy_xx - f)
        self.add_equation("res2", -dy_xxx - df_x)


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    invar: str = cfg.MODEL.input_keys[0]
    outvar: str = cfg.MODEL.output_keys[0]

    def output_transform(
        in_: Dict[str, paddle.Tensor], out: Dict[str, paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        x = in_[invar]
        u = out[outvar]
        return {
            outvar: x + paddle.tanh(x) * paddle.tanh(np.pi - x) * u,
        }

    model.register_output_transform(output_transform)

    # set equation
    equation = {"gPINN": gPINN1D(invar, outvar)}

    # set geometry
    geom = {"line": ppsci.geometry.Interval(0, np.pi)}

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["gPINN"].equations,
        {"res1": 0, "res2": 0},
        geom["line"],
        {**train_dataloader_cfg, "batch_size": cfg.NPOINT_PDE},
        ppsci.loss.MSELoss("mean", weight={"res2": 0.01}),
        evenly=True,
        name="EQ",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(0.001)(model)

    # set validator
    def u_solution(in_):
        x = in_[invar]
        sol = x + 1 / 8 * np.sin(8 * x)
        for i in range(1, 5):
            sol += 1 / i * np.sin(i * x)
        return sol

    l2rel_validator = ppsci.validate.GeometryValidator(
        {outvar: lambda out: out[outvar]},
        {outvar: u_solution},
        geom["line"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": cfg.NPOINT_PDE_EVAL,
            "batch_size": cfg.EVAL.batch_size.l2rel_validator,
            "sampler": {"name": "BatchSampler"},
        },
        ppsci.loss.MSELoss("mean"),
        evenly=True,
        metric={f"L2Rel({outvar})": ppsci.metric.L2Rel()},
        name="L2Rel",
    )
    validator = {l2rel_validator.name: l2rel_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # visualize prediction for outvar
    x = geom["line"].uniform_points(1000)
    plt.figure()
    plt.plot(x, u_solution({invar: x}), label="Exact", color="black")
    plt.plot(
        x,
        solver.predict({invar: x}, return_numpy=True)[outvar],
        label="gPINN, w = 0.01",
        color="red",
        linestyle="dashed",
    )
    plt.legend(frameon=False)
    plt.xlabel(invar)
    plt.ylabel(outvar)

    x = geom["line"].uniform_points(15, boundary=False)
    plt.plot(x, u_solution({invar: x}), color="black", marker="o", linestyle="none")
    # save visualization result for prediction of outvar
    plt.savefig(osp.join(cfg.output_dir, f"pred_{outvar}.png"))
    plt.clf()

    # visualize prediction for du/dx
    x = geom["line"].uniform_points(1000)
    plt.figure()

    def du_x(x: np.ndarray) -> np.ndarray:
        return (
            1
            + np.cos(x)
            + np.cos(2 * x)
            + np.cos(3 * x)
            + np.cos(4 * x)
            + np.cos(8 * x)
        )

    plt.plot(x, du_x(x), label="Exact", color="black")
    plt.plot(
        x,
        solver.predict(
            {invar: x},
            return_numpy=True,
            expr_dict={
                f"d{outvar}d{invar}": lambda out: jacobian(out[outvar], out[invar])
            },
            no_grad=False,
        )[f"d{outvar}d{invar}"],
        label="gPINN, w = 0.01",
        color="red",
        linestyle="dashed",
    )
    x = geom["line"].uniform_points(15, boundary=False)
    plt.plot(x, du_x(x), color="black", marker="o", linestyle="none")
    plt.legend(frameon=False)
    plt.xlabel(invar)
    plt.ylabel(outvar)
    # save visualization result of prediction 'du/dx'
    plt.savefig(osp.join(cfg.output_dir, f"pred_d{outvar}d{invar}.png"))


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    invar: str = cfg.MODEL.input_keys[0]
    outvar: str = cfg.MODEL.output_keys[0]

    def output_transform(in_, out):
        x = in_[invar]
        u = out[outvar]
        return {
            outvar: x + paddle.tanh(x) * paddle.tanh(np.pi - x) * u,
        }

    model.register_output_transform(output_transform)

    # set geometry
    geom = {"line": ppsci.geometry.Interval(0, np.pi)}

    # set validator
    def u_solution(in_):
        x = in_[invar]
        sol = x + 1 / 8 * np.sin(8 * x)
        for i in range(1, 5):
            sol += 1 / i * np.sin(i * x)
        return sol

    l2rel_validator = ppsci.validate.GeometryValidator(
        {outvar: lambda out: out[outvar]},
        {outvar: u_solution},
        geom["line"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": cfg.NPOINT_PDE,
            "batch_size": cfg.EVAL.batch_size.l2rel_validator,
            "sampler": {"name": "BatchSampler"},
        },
        ppsci.loss.MSELoss("mean"),
        evenly=True,
        metric={f"L2Rel({outvar})": ppsci.metric.L2Rel()},
        name="L2Rel",
    )
    validator = {l2rel_validator.name: l2rel_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    # evaluate after finished training
    solver.eval()

    # visualize prediction for outvar
    x = geom["line"].uniform_points(1000)
    plt.figure()
    plt.plot(x, u_solution({invar: x}), label="Exact", color="black")
    plt.plot(
        x,
        solver.predict({invar: x}, return_numpy=True)[outvar],
        label="gPINN, w = 0.01",
        color="red",
        linestyle="dashed",
    )
    plt.legend(frameon=False)
    plt.xlabel(invar)
    plt.ylabel(outvar)

    x = geom["line"].uniform_points(15, boundary=False)
    plt.plot(x, u_solution({invar: x}), color="black", marker="o", linestyle="none")
    # save visualization result for prediction of outvar
    plt.savefig(osp.join(cfg.output_dir, f"pred_{outvar}.png"))
    plt.clf()

    # visualize prediction for du/dx
    x = geom["line"].uniform_points(1000)
    plt.figure()

    def du_x(x):
        return (
            1
            + np.cos(x)
            + np.cos(2 * x)
            + np.cos(3 * x)
            + np.cos(4 * x)
            + np.cos(8 * x)
        )

    plt.plot(x, du_x(x), label="Exact", color="black")
    plt.plot(
        x,
        solver.predict(
            {invar: x},
            return_numpy=True,
            expr_dict={
                f"d{outvar}d{invar}": lambda out: jacobian(out[outvar], out[invar])
            },
            no_grad=False,
        )[f"d{outvar}d{invar}"],
        label="gPINN, w = 0.01",
        color="red",
        linestyle="dashed",
    )
    x = geom["line"].uniform_points(15, boundary=False)
    plt.plot(x, du_x(x), color="black", marker="o", linestyle="none")
    plt.legend(frameon=False)
    plt.xlabel(invar)
    plt.ylabel(outvar)
    # save visualization result of prediction 'du/dx'
    plt.savefig(osp.join(cfg.output_dir, f"pred_d{outvar}d{invar}.png"))


@hydra.main(version_base=None, config_path="./conf", config_name="poisson_1d.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

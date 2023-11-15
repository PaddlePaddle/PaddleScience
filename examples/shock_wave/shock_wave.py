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
import lhs
import numpy as np
import paddle
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci import equation
from ppsci.autodiff import jacobian
from ppsci.utils import logger
from ppsci.utils import misc


class Euler2D(equation.PDE):
    def __init__(self):
        super().__init__()
        # HACK: solver will be added here for tracking run-time epoch to
        # compute loss factor `relu` dynamically.
        self.solver: ppsci.solver.Solver = None

        def continuity_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            t, x, y = out["t"], out["x"], out["y"]
            u, v, rho = out["u"], out["v"], out["rho"]
            rho__t = jacobian(rho, t)
            rho_u = rho * u
            rho_v = rho * v
            rho_u__x = jacobian(rho_u, x)
            rho_v__y = jacobian(rho_v, y)

            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            delta_u = u__x + v__y
            nab = paddle.abs(delta_u) - delta_u
            lam = (0.1 * nab) * relu + 1
            continuity = (rho__t + rho_u__x + rho_v__y) / lam
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def x_momentum_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            t, x, y = out["t"], out["x"], out["y"]
            u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
            rho_u = rho * u
            rho_u__t = jacobian(rho_u, t)

            u1 = rho * u**2 + p
            u2 = rho * u * v
            u1__x = jacobian(u1, x)
            u2__y = jacobian(u2, y)

            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            delta_u = u__x + v__y
            nab = paddle.abs(delta_u) - delta_u
            lam = (0.1 * nab) * relu + 1
            x_momentum = (rho_u__t + u1__x + u2__y) / lam
            return x_momentum

        self.add_equation("x_momentum", x_momentum_compute_func)

        def y_momentum_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            t, x, y = out["t"], out["x"], out["y"]
            u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
            rho_v = rho * v
            rho_v__t = jacobian(rho_v, t)

            u2 = rho * u * v
            u3 = rho * v**2 + p
            u2__x = jacobian(u2, x)
            u3__y = jacobian(u3, y)

            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            delta_u = u__x + v__y
            nab = paddle.abs(delta_u) - delta_u
            lam = (0.1 * nab) * relu + 1
            y_momentum = (rho_v__t + u2__x + u3__y) / lam
            return y_momentum

        self.add_equation("y_momentum", y_momentum_compute_func)

        def energy_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            t, x, y = out["t"], out["x"], out["y"]
            u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
            e1 = (rho * 0.5 * (u**2 + v**2) + 3.5 * p) * u
            e2 = (rho * 0.5 * (u**2 + v**2) + 3.5 * p) * v
            e = rho * 0.5 * (u**2 + v**2) + p / 0.4

            e1__x = jacobian(e1, x)
            e2__y = jacobian(e2, y)
            e__t = jacobian(e, t)

            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            delta_u = u__x + v__y
            nab = paddle.abs(delta_u) - delta_u
            lam = (0.1 * nab) * relu + 1
            energy = (e__t + e1__x + e2__y) / lam
            return energy

        self.add_equation("energy", energy_compute_func)


class BC_EQ(equation.PDE):
    def __init__(self):
        super().__init__()
        # HACK: solver will be added here for tracking run-time epoch to
        # compute loss factor `relu` dynamically.
        self.solver: ppsci.solver.Solver = None

        def item1_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            sin, cos = out["sin"], out["cos"]
            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            delta_u = u__x + v__y

            lam = 0.1 * (paddle.abs(delta_u) - delta_u) * relu + 1
            item1 = (u * cos + v * sin) / lam

            return item1

        self.add_equation("item1", item1_compute_func)

        def item2_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            sin, cos = out["sin"], out["cos"]
            p__x = jacobian(p, x)
            p__y = jacobian(p, y)
            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            delta_u = u__x + v__y

            lam = 0.1 * (paddle.abs(delta_u) - delta_u) * relu + 1
            item2 = (p__x * cos + p__y * sin) / lam

            return item2

        self.add_equation("item2", item2_compute_func)

        def item3_compute_func(out):
            relu = max(
                0.0,
                (self.solver.global_step // self.solver.iters_per_epoch + 1)
                / self.solver.epochs
                - 0.05,
            )
            x, y = out["x"], out["y"]
            u, v, rho = out["u"], out["v"], out["rho"]
            sin, cos = out["sin"], out["cos"]
            u__x = jacobian(u, x)
            v__y = jacobian(v, y)
            rho__x = jacobian(rho, x)
            rho__y = jacobian(rho, y)
            delta_u = u__x + v__y

            lam = 0.1 * (paddle.abs(delta_u) - delta_u) * relu + 1
            item3 = (rho__x * cos + rho__y * sin) / lam

            return item3

        self.add_equation("item3", item3_compute_func)


dtype = paddle.get_default_dtype()


def generate_bc_down_circle_points(t: float, xc: float, yc: float, r: float, n: int):
    rand_arr1 = np.random.randn(n, 1).astype(dtype)
    theta = 2 * np.pi * rand_arr1
    cos = np.cos(np.pi / 2 + theta)
    sin = np.sin(np.pi / 2 + theta)

    rand_arr2 = np.random.randn(n, 1).astype(dtype)
    x = np.concatenate([rand_arr2 * t, xc + cos * r, yc + sin * r], axis=1)

    return x, sin, cos


def generate_bc_left_points(
    x: np.ndarray, Ma: float, rho1: float, p1: float, v1: float, gamma: float
):
    u1: float = np.sqrt(gamma * p1 / rho1) * Ma
    u_init = np.full((x.shape[0], 1), u1, dtype)
    v_init = np.full((x.shape[0], 1), v1, dtype)
    p_init = np.full((x.shape[0], 1), p1, dtype)
    rho_init = np.full((x.shape[0], 1), rho1, dtype)

    return u_init, v_init, p_init, rho_init


def train(cfg: DictConfig):

    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"Euler2D": Euler2D(), "BC_EQ": BC_EQ()}

    # Latin HyperCube Sampling
    # generate PDE data
    xlimits = np.array([[0.0, 0.0, 0.0], [cfg.Lt, cfg.Lx, cfg.Ly]]).T
    doe_lhs = lhs.LHS(cfg.N_INTERIOR, xlimits)
    x_int_train = doe_lhs.get_sample()
    x_int_train = x_int_train[
        ~(
            (x_int_train[:, 1] - cfg.rx) ** 2 + (x_int_train[:, 2] - cfg.ry) ** 2
            < cfg.rd**2
        )
    ]
    x_int_train_dict = misc.convert_to_dict(x_int_train, cfg.MODEL.input_keys)

    y_int_train = np.zeros([len(x_int_train), len(cfg.MODEL.output_keys)], dtype)
    y_int_train_dict = misc.convert_to_dict(
        y_int_train, tuple(equation["Euler2D"].equations.keys())
    )

    # generate BC data(left, right side)
    xlimits = np.array([[0.0, 0.0, 0.0], [cfg.Lt, 0.0, cfg.Ly]]).T
    doe_lhs = lhs.LHS(cfg.N_BOUNDARY, xlimits)
    x_bcL_train = doe_lhs.get_sample()
    x_bcL_train_dict = misc.convert_to_dict(x_bcL_train, cfg.MODEL.input_keys)

    u_bcL_train, v_bcL_train, p_bcL_train, rho_bcL_train = generate_bc_left_points(
        x_bcL_train, cfg.MA, cfg.RHO1, cfg.P1, cfg.V1, cfg.GAMMA
    )
    y_bcL_train = np.concatenate(
        [
            u_bcL_train,
            v_bcL_train,
            p_bcL_train,
            rho_bcL_train,
        ],
        axis=1,
    )
    y_bcL_train_dict = misc.convert_to_dict(
        y_bcL_train,
        tuple(model.output_keys),
    )

    x_bcI_train, sin_bcI_train, cos_bcI_train = generate_bc_down_circle_points(
        cfg.Lt, cfg.rx, cfg.ry, cfg.rd, cfg.N_BOUNDARY
    )
    x_bcI_train_dict = misc.convert_to_dict(
        np.concatenate([x_bcI_train, sin_bcI_train, cos_bcI_train], axis=1),
        cfg.MODEL.input_keys + ["sin", "cos"],
    )
    y_bcI_train_dict = misc.convert_to_dict(
        np.zeros((len(x_bcI_train), 3), dtype),
        ("item1", "item2", "item3"),
    )

    # generate IC data
    xlimits = np.array([[0.0, 0.0, 0.0], [0.0, cfg.Lx, cfg.Ly]]).T
    doe_lhs = lhs.LHS(cfg.N_BOUNDARY, xlimits)
    x_ic_train = doe_lhs.get_sample()
    x_ic_train = x_ic_train[
        ~(
            (x_ic_train[:, 1] - cfg.rx) ** 2 + (x_ic_train[:, 2] - cfg.ry) ** 2
            < cfg.rd**2
        )
    ]
    x_ic_train_dict = misc.convert_to_dict(x_ic_train, cfg.MODEL.input_keys)
    U1 = np.sqrt(cfg.GAMMA * cfg.P1 / cfg.RHO1) * cfg.MA
    y_ic_train = np.concatenate(
        [
            np.full([len(x_ic_train), 1], U1, dtype),
            np.full([len(x_ic_train), 1], 0, dtype),
            np.full([len(x_ic_train), 1], cfg.P1, dtype),
            np.full([len(x_ic_train), 1], cfg.RHO1, dtype),
        ],
        axis=1,
    )
    y_ic_train_dict = misc.convert_to_dict(
        y_ic_train,
        model.output_keys,
    )

    # set constraints
    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_int_train_dict,
                "label": y_int_train_dict,
            },
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr=equation["Euler2D"].equations,
        name="PDE",
    )
    ic_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_ic_train_dict,
                "label": y_ic_train_dict,
            },
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean", weight=10),
        name="IC",
    )
    bcI_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_bcI_train_dict,
                "label": y_bcI_train_dict,
            },
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean", weight=10),
        output_expr=equation["BC_EQ"].equations,
        name="BCI",
    )
    bcL_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_bcL_train_dict,
                "label": y_bcL_train_dict,
            },
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean", weight=10),
        name="BCL",
    )
    constraint = {
        pde_constraint.name: pde_constraint,
        ic_constraint.name: ic_constraint,
        bcI_constraint.name: bcI_constraint,
        bcL_constraint.name: bcL_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.LBFGS(
        cfg.TRAIN.learning_rate, max_iter=cfg.TRAIN.max_iter
    )(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        equation=equation,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # HACK: Given entire solver to euaqtion object for tracking run-time epoch
    # to compute factor `relu` dynamically.
    equation["Euler2D"].solver = solver
    equation["BC_EQ"].solver = solver

    # train model
    solver.train()


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )

    # visualize prediction
    t = np.linspace(cfg.T, cfg.T, 1)
    x = np.linspace(0.0, cfg.Lx, cfg.Nd)
    y = np.linspace(0.0, cfg.Ly, cfg.Nd)
    _, x_grid, y_grid = np.meshgrid(t, x, y)

    x_test = misc.cartesian_product(t, x, y)
    x_test_dict = misc.convert_to_dict(
        x_test,
        cfg.MODEL.input_keys,
    )

    output_dict = solver.predict(x_test_dict, return_numpy=True)
    u, v, p, rho = (
        output_dict["u"],
        output_dict["v"],
        output_dict["p"],
        output_dict["rho"],
    )

    zero_mask = (
        (x_test[:, 1] - cfg.rx) ** 2 + (x_test[:, 2] - cfg.ry) ** 2
    ) < cfg.rd**2
    u[zero_mask] = 0
    v[zero_mask] = 0
    p[zero_mask] = 0
    rho[zero_mask] = 0

    u = u.reshape(cfg.Nd, cfg.Nd)
    v = v.reshape(cfg.Nd, cfg.Nd)
    p = p.reshape(cfg.Nd, cfg.Nd)
    rho = rho.reshape(cfg.Nd, cfg.Nd)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], u * 241.315, 60)
    plt.title("U m/s")
    plt.xlabel("x")
    plt.ylabel("y")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], v * 241.315, 60)
    plt.title("V m/s")
    plt.xlabel("x")
    plt.ylabel("y")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], p * 33775, 60)
    plt.title("P Pa")
    plt.xlabel("x")
    plt.ylabel("y")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], rho * 0.58, 60)
    plt.title("Rho kg/m^3")
    plt.xlabel("x")
    plt.ylabel("y")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.savefig(osp.join(cfg.output_dir, f"shock_wave(Ma_{cfg.MA:.3f}).png"))


@hydra.main(
    version_base=None, config_path="./conf", config_name="shock_wave_Ma2.0.yaml"
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

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

from os import path

import lhs
import numpy as np
import paddle
from matplotlib import pyplot as plt

import ppsci
from ppsci import equation
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import misc


class Euler2D(equation.PDE):
    def __init__(self, total_epoch: int):
        super().__init__()
        self.total_epoch = total_epoch
        self.epoch = 0

        def continuity_compute_func(out):
            self.epoch += 1
            relu = 0
            t, x, y = out["t"], out["x"], out["y"]
            u, v, rho = out["u"], out["v"], out["rho"]
            drho_t = jacobian(rho, t)
            rhou = rho * u
            rhov = rho * v
            drhou_x = jacobian(rhou, x)
            drhov_y = jacobian(rhov, y)

            u_x = jacobian(u, x)
            v_y = jacobian(v, y)
            deltaU = u_x + v_y
            nab = paddle.abs(deltaU) - deltaU
            lam = (0.1 * nab) * relu + 1
            continuity = (drho_t + drhou_x + drhov_y) / lam
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def x_momentum_compute_func(out):
            relu = 0
            t, x, y = out["t"], out["x"], out["y"]
            u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
            rhou = rho * u
            drhou_t = jacobian(rhou, t)

            U1 = rho * u**2 + p
            U2 = rho * u * v
            dU1_x = jacobian(U1, x)
            dU2_y = jacobian(U2, y)

            u_x = jacobian(u, x)
            v_y = jacobian(v, y)
            deltaU = u_x + v_y
            nab = paddle.abs(deltaU) - deltaU
            lam = (0.1 * nab) * relu + 1
            x_momentum = (drhou_t + dU1_x + dU2_y) / lam
            return x_momentum

        self.add_equation("x_momentum", x_momentum_compute_func)

        def y_momentum_compute_func(out):
            relu = 0
            t, x, y = out["t"], out["x"], out["y"]
            u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
            rhov = rho * v
            drhov_t = jacobian(rhov, t)

            U2 = rho * u * v
            U3 = rho * v**2 + p
            dU2_x = jacobian(U2, x)
            dU3_y = jacobian(U3, y)

            u_x = jacobian(u, x)
            v_y = jacobian(v, y)
            deltaU = u_x + v_y
            nab = paddle.abs(deltaU) - deltaU
            lam = (0.1 * nab) * relu + 1
            y_momentum = (drhov_t + dU2_x + dU3_y) / lam
            return y_momentum

        self.add_equation("y_momentum", y_momentum_compute_func)

        def energy_compute_func(out):
            relu = 0
            t, x, y = out["t"], out["x"], out["y"]
            u, v, p, rho = out["u"], out["v"], out["p"], out["rho"]
            E1 = (rho * 0.5 * (u**2 + v**2) + 3.5 * p) * u
            E2 = (rho * 0.5 * (u**2 + v**2) + 3.5 * p) * v
            E = rho * 0.5 * (u**2 + v**2) + p / 0.4

            dE1_x = jacobian(E1, x)
            dE2_y = jacobian(E2, y)
            dE_t = jacobian(E, t)

            u_x = jacobian(u, x)
            v_y = jacobian(v, y)
            deltaU = u_x + v_y
            nab = paddle.abs(deltaU) - deltaU
            lam = (0.1 * nab) * relu + 1
            energy = (dE_t + dE1_x + dE2_y) / lam
            return energy

        self.add_equation("energy", energy_compute_func)


class BC_W(equation.PDE):
    def __init__(self, total_epoch: int):
        super().__init__()
        self.total_epoch = total_epoch
        self.epoch = 0

        def item1_compute_func(out):
            self.epoch += 1
            relu = 0
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            sin, cos = out["sin"], out["cos"]
            dudx = jacobian(u, x)
            dvdy = jacobian(v, y)
            deltau = dudx + dvdy

            lam = 0.1 * (paddle.abs(deltau) - deltau) * relu + 1
            item1 = (u * cos + v * sin) / lam

            return item1

        self.add_equation("item1", item1_compute_func)

        def item2_compute_func(out):
            relu = 0
            x, y = out["x"], out["y"]
            u, v, p = out["u"], out["v"], out["p"]
            sin, cos = out["sin"], out["cos"]
            dpdx = jacobian(p, x)
            dpdy = jacobian(p, y)
            dudx = jacobian(u, x)
            dvdy = jacobian(v, y)
            deltau = dudx + dvdy

            lam = 0.1 * (paddle.abs(deltau) - deltau) * relu + 1
            item2 = (dpdx * cos + dpdy * sin) / lam

            return item2

        self.add_equation("item2", item2_compute_func)

        def item3_compute_func(out):
            relu = 0
            x, y = out["x"], out["y"]
            u, v, rho = out["u"], out["v"], out["rho"]
            sin, cos = out["sin"], out["cos"]
            dudx = jacobian(u, x)
            dvdy = jacobian(v, y)
            drhodx = jacobian(rho, x)
            drhody = jacobian(rho, y)
            deltau = dudx + dvdy

            lam = 0.1 * (paddle.abs(deltau) - deltau) * relu + 1
            item3 = (drhodx * cos + drhody * sin) / lam

            return item3

        self.add_equation("item3", item3_compute_func)


def BD_circle(t, xc, yc, r, n):
    x = np.zeros((n, 3), paddle.get_default_dtype())
    sin = np.zeros((n, 1), paddle.get_default_dtype())
    cos = np.zeros((n, 1), paddle.get_default_dtype())

    for i in range(n):
        the = 2 * np.random.rand() * np.pi
        xd = np.cos(the + np.pi / 2)
        yd = np.sin(the + np.pi / 2)
        x[i, 2] = np.random.rand() * t
        x[i, 0] = xc + xd * r
        x[i, 1] = yc + yd * r
        cos[i, 0] = xd
        sin[i, 0] = yd
        # cos[i,0] = 1
        # sin[i,0] = 0
    return x, sin, cos


def BC_L(x, Ma, rho1, p1, v1, gamma):
    N = x.shape[0]
    rho_init = np.zeros((x.shape[0], 1), paddle.get_default_dtype())
    u_init = np.zeros((x.shape[0], 1), paddle.get_default_dtype())
    v_init = np.zeros((x.shape[0], 1), paddle.get_default_dtype())
    p_init = np.zeros((x.shape[0], 1), paddle.get_default_dtype())

    # gamma = 1.4
    # rho1 = 2.112
    # p1 = 3.001
    # v1 = 0.0
    u1 = np.sqrt(gamma * p1 / rho1) * Ma

    for i in range(N):
        rho_init[i] = rho1
        u_init[i] = u1
        v_init[i] = v1
        p_init[i] = p1

    return rho_init, u_init, v_init, p_init


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    SEED = 42
    MA = 2.0
    ppsci.utils.misc.set_random_seed(SEED)
    # set output directory
    OUTPUT_DIR = (
        f"./output_pinn_we_{MA:.3f}" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(("x", "y", "t"), ("rho", "p", "u", "v"), 9, 90, "tanh")

    # load random ckpt
    init_sd = paddle.load("Net.pdparams")
    proc_sd = {}
    layer_num = 0
    for i, (k, v) in enumerate(init_sd.items()):
        if i % 2 == 0:
            if i >= 18:
                proc_sd["last_fc.weight"] = v
            else:
                proc_sd[f"linears.{layer_num}.weight"] = v
        else:
            if i >= 18:
                proc_sd["last_fc.bias"] = v
            else:
                proc_sd[f"linears.{layer_num}.bias"] = v
            layer_num += 1
    model.load_dict(proc_sd)

    # set equation
    EPOCHS = 100
    equation = {"Euler2D": Euler2D(EPOCHS), "BC_W": BC_W(EPOCHS)}

    # set dataloader config
    ITERS_PER_EPOCH = 1

    # num_ib = 30000  # Random sampled points from IC,BC
    num_ib = 10000
    # num_int = 200000  # Random sampled points in interior
    num_int = 100000
    Tend = 0.4
    Lx = 1.5
    Ly = 2.0
    rx = 1.0
    ry = 1.0
    rd = 0.25
    # Latin HyperCube Sampling
    xlimits = np.array([[0.0, 0.0, 0.0], [Lx, Ly, Tend]]).T
    name_value = ("x", "y", "t")
    doe_lhs = lhs.DoE_LHS(name_value, num_int, xlimits)
    x_int_train = doe_lhs.get_sample()
    x_int_train = x_int_train[
        ~((x_int_train[:, 0] - rx) ** 2 + (x_int_train[:, 1] - ry) ** 2 < rd**2)
    ]
    x_int_train_dict = misc.convert_to_dict(x_int_train, name_value)

    y_int_train = np.zeros(
        [len(x_int_train), len(model.output_keys)], paddle.get_default_dtype()
    )
    y_int_train_dict = misc.convert_to_dict(
        y_int_train, tuple(equation["Euler2D"].equations.keys())
    )

    # initial conditions
    xlimits = np.array([[0.0, 0.0, 0.0], [Lx, Ly, 0.0]]).T
    doe_lhs = lhs.DoE_LHS(name_value, num_ib, xlimits)
    x_ic_train = doe_lhs.get_sample()
    x_ic_train = x_ic_train[
        ~((x_ic_train[:, 0] - rx) ** 2 + (x_ic_train[:, 1] - ry) ** 2 < rd**2)
    ]
    x_ic_train_dict = misc.convert_to_dict(x_ic_train, name_value)

    # set hyper-parameters
    RHO1 = 2.112
    P1 = 3.001
    GAMMA = 1.4
    V1 = 0.0
    U1 = np.sqrt(GAMMA * P1 / RHO1) * MA
    y_ic_train = np.concatenate(
        [
            np.full([len(x_ic_train), 1], RHO1, paddle.get_default_dtype()),
            np.full([len(x_ic_train), 1], P1, paddle.get_default_dtype()),
            np.full([len(x_ic_train), 1], U1, paddle.get_default_dtype()),
            np.full([len(x_ic_train), 1], 0, paddle.get_default_dtype()),
        ],
        axis=1,
    )
    y_ic_train_dict = misc.convert_to_dict(
        y_ic_train,
        model.output_keys,
    )

    # 1、bound field(left, right side)
    xlimits = np.array(
        [
            [0.0, 0.0, 0.0],
            [
                0.0,
                Ly,
                Tend,
            ],
        ]
    ).T
    name_value = ("x", "y", "t")
    doe_lhs = lhs.DoE_LHS(name_value, num_ib, xlimits)
    x_bcL_train = doe_lhs.get_sample()
    x_bcL_train_dict = misc.convert_to_dict(x_bcL_train, name_value)

    rho_bcL_train, u_bcL_train, v_bcL_train, p_bcL_train = BC_L(
        x_bcL_train, MA, RHO1, P1, V1, GAMMA
    )
    y_bcL_train = np.concatenate(
        [
            rho_bcL_train,
            p_bcL_train,
            u_bcL_train,
            v_bcL_train,
        ],
        axis=1,
    )
    y_bcL_train_dict = misc.convert_to_dict(
        y_bcL_train,
        tuple(model.output_keys),
    )

    x_bcI_train, sin_bcI_train, cos_bcI_train = BD_circle(Tend, rx, ry, rd, num_ib)
    x_bcI_train_dict = misc.convert_to_dict(
        np.concatenate([x_bcI_train, sin_bcI_train, cos_bcI_train], axis=1),
        name_value + ("sin", "cos"),
    )
    y_bcI_train_dict = misc.convert_to_dict(
        np.zeros((len(x_bcI_train), 3), paddle.get_default_dtype()),
        ("item1", "item2", "item3"),
    )

    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_int_train_dict,
                "label": y_int_train_dict,
            },
            "iters_per_epoch": ITERS_PER_EPOCH,
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
            "iters_per_epoch": ITERS_PER_EPOCH,
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
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean", weight=10),
        output_expr=equation["BC_W"].equations,
        name="BCI",
    )
    bcL_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": x_bcL_train_dict,
                "label": y_bcL_train_dict,
            },
            "iters_per_epoch": ITERS_PER_EPOCH,
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
    optimizer = ppsci.optimizer.LBFGS(1e-1, max_iter=100)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        log_freq=20,
        eval_during_train=False,
        seed=SEED,
        equation=equation,
        eval_with_no_grad=True,
        # pretrained_model_path=path.join(OUTPUT_DIR, "checkpoints", "latest")
    )
    """
    epoch1, total loss : 396.18261719, loss pde : 0.40235952,
    loss boundary conditions : 0.05253036,  23.73625755, loss initial conditions : 15.78923607
    """
    # train model
    solver.train()

    # visualize prediction
    Nd = 600
    T = 0.4
    t = np.linspace(T, T, 1)
    x = np.linspace(0.0, Lx, Nd)
    y = np.linspace(0.0, Ly, Nd)
    t_grid, x_grid, y_grid = np.meshgrid(t, x, y)

    x_test = misc.cartesian_product(x, y, t)
    x_test_dict = misc.convert_to_dict(
        x_test,
        name_value,
    )
    output_dict = solver.predict(x_test_dict, return_numpy=True)
    zero_mask = ((x_test[:, 0] - rx) ** 2 + (x_test[:, 1] - ry) ** 2) < rd**2

    rho, p, u, v = (
        output_dict["rho"],
        output_dict["p"],
        output_dict["u"],
        output_dict["v"],
    )

    rho[zero_mask] = 0
    p[zero_mask] = 0
    u[zero_mask] = 0
    v[zero_mask] = 0

    rho = rho.reshape(Nd, Nd)
    p = p.reshape(Nd, Nd)
    u = u.reshape(Nd, Nd)
    v = v.reshape(Nd, Nd)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], rho * 0.58, 60)
    plt.title("Rho kg/m^3")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], p * 33775, 60)
    plt.title("P Pa")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], u * 241.315, 60)
    plt.title("U m/s")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.contourf(x_grid[:, 0, :], y_grid[:, 0, :], v * 241.315, 60)
    plt.title("V m/s")
    axe = plt.gca()
    axe.set_aspect(1)
    plt.colorbar()

    plt.savefig(path.join(OUTPUT_DIR, f"shock_result(Ma_{MA:.3f}).png"))

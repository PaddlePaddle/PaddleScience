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

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
import seaborn as sns
from paddle.fluid import core

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base

if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)

    # set output directory
    output_dir = "./output_poiseuille_flow"

    core.set_prim_eager_enabled(False)

    # initialize logger
    ppsci.utils.logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    NU_MEAN = 0.001
    NU_STD = 0.9
    L = 1.0  # length of pipe
    R = 0.05  # radius of pipe
    RHO = 1  # density
    P_OUT = 0  # pressure at the outlet of pipe
    P_IN = 0.1  # pressure at the inlet of pipe

    N_x = 10
    N_y = 50
    N_p = 50

    LEARNING_RATE = 5e-3
    BATCH_SIZE = 128
    EPOCHS = 3000  # 5000
    ITERS_PER_EPOCH = int((N_x * N_y * N_p) / BATCH_SIZE)

    EVAL_FREQ = 100  # display step
    VISU_FREQ = 100  # visulize step

    X_IN = 0
    X_OUT = X_IN + L
    Y_START = -R
    Y_END = Y_START + 2 * R
    NU_START = NU_MEAN - NU_MEAN * NU_STD  # 0.0001
    NU_END = NU_MEAN + NU_MEAN * NU_STD  # 0.1

    ## prepare data with (?, 2)
    data_1d_x = np.linspace(
        X_IN, X_OUT, N_x, endpoint=True, dtype=paddle.get_default_dtype()
    )
    data_1d_y = np.linspace(
        Y_START, Y_END, N_y, endpoint=True, dtype=paddle.get_default_dtype()
    )
    data_1d_nu = np.linspace(
        NU_START, NU_END, N_p, endpoint=True, dtype=paddle.get_default_dtype()
    )

    data_2d_xy = (
        np.array(np.meshgrid(data_1d_x, data_1d_y, data_1d_nu)).reshape(3, -1).T
    )
    data_2d_xy_old = copy.deepcopy(data_2d_xy)
    np.random.shuffle(data_2d_xy)

    input_x = data_2d_xy[:, 0].reshape(data_2d_xy.shape[0], 1)
    input_y = data_2d_xy[:, 1].reshape(data_2d_xy.shape[0], 1)
    input_nu = data_2d_xy[:, 2].reshape(data_2d_xy.shape[0], 1)

    interior_data = {"x": input_x, "y": input_y, "nu": input_nu}
    interior_geom = ppsci.geometry.PointCloud(
        interior={"x": input_x, "y": input_y, "nu": input_nu},
        coord_keys=("x", "y", "nu"),
    )

    # set model
    model_u = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("u"), 3, 50, "swish")
    model_v = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("v"), 3, 50, "swish")
    model_p = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("p"), 3, 50, "swish")

    class Transform:
        def __init__(self) -> None:
            pass

        def input_trans(self, input):
            self.input = input
            x, y = input["x"], input["y"]
            nu = input["nu"]
            b = 2 * np.pi / (X_OUT - X_IN)
            c = np.pi * (X_IN + X_OUT) / (X_IN - X_OUT)
            sin_x = X_IN * paddle.sin(b * x + c)
            cos_x = X_IN * paddle.cos(b * x + c)
            return {"sin(x)": sin_x, "cos(x)": cos_x, "y": y, "nu": nu}

        def output_trans_u(self, out):
            return {"u": out["u"] * (R**2 - self.input["y"] ** 2)}

        def output_trans_v(self, out):
            return {"v": (R**2 - self.input["y"] ** 2) * out["v"]}

        def output_trans_p(self, out):
            return {
                "p": (
                    (X_IN - self.input["x"]) * 0
                    + (P_IN - P_OUT) * (X_OUT - self.input["x"]) / L
                    + (X_IN - self.input["x"]) * (X_OUT - self.input["x"]) * out["p"]
                )
            }

    model_u.register_input_transform(Transform.input_trans)
    model_v.register_input_transform(Transform.input_trans)
    model_p.register_input_transform(Transform.input_trans)
    model_u.register_output_transform(Transform.output_trans_u)
    model_v.register_output_transform(Transform.output_trans_v)
    model_p.register_output_transform(Transform.output_trans_p)
    model = ppsci.arch.ModelList((model_u, model_v, model_p))

    # set optimizer
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE)((model,))

    # set euqation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu=lambda out: out["nu"], rho=RHO, dim=2, time=False
        )
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom=interior_geom,
        dataloader_cfg={
            "dataset": "NamedArrayDataset",
            "num_workers": 1,
            "batch_size": BATCH_SIZE,
            "iters_per_epoch": ITERS_PER_EPOCH,
            "sampler": {
                "name": "BatchSampler",
                "shuffle": False,
                "drop_last": False,
            },
        },
        loss=ppsci.loss.MSELoss("mean"),
        evenly=True,
        name="EQ",
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        {pde_constraint.name: pde_constraint},
        output_dir,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        save_freq=10,
        log_freq=1,
        equation=equation,
    )

    solver.train()

    # Cross-section velocity profiles of 4 different viscosity sample
    # Predicted result
    input_dict = {
        "x": data_2d_xy_old[:, 0:1],
        "y": data_2d_xy_old[:, 1:2],
        "nu": data_2d_xy_old[:, 2:3],
    }
    output_dict = solver.predict(input_dict)
    u_pred = output_dict["u"].numpy().reshape(N_y, N_x, N_p)
    v_pred = output_dict["v"].numpy().reshape(N_y, N_x, N_p)
    p_pred = output_dict["p"].numpy().reshape(N_y, N_x, N_p)

    # Analytical result, y = data_1d_y
    u_analytical = np.zeros([N_y, N_x, N_p])
    dP = P_IN - P_OUT

    for i in range(N_p):
        uy = (R**2 - data_1d_y**2) * dP / (2 * L * data_1d_nu[i] * RHO)
        u_analytical[:, :, i] = np.tile(uy.reshape([N_y, 1]), N_x)

    fontsize = 16
    idx_X = int(round(N_x / 2))  # pipe velocity section at L/2
    nu_index = [3, 6, 14, 49]  # pick 4 nu samples
    ytext = [0.45, 0.28, 0.1, 0.01]

    # Plot
    plt.figure(1)
    plt.clf()
    for idxP in range(len(nu_index)):
        ax1 = plt.subplot(111)
        plt.plot(
            data_1d_y,
            u_analytical[:, idx_X, nu_index[idxP]],
            color="darkblue",
            linestyle="-",
            lw=3.0,
            alpha=1.0,
        )
        plt.plot(
            data_1d_y,
            u_pred[:, idx_X, nu_index[idxP]],
            color="red",
            linestyle="--",
            dashes=(5, 5),
            lw=2.0,
            alpha=1.0,
        )
        nu_current = float("{0:.5f}".format(data_1d_nu[nu_index[idxP]]))
        plt.text(
            -0.012,
            ytext[idxP],
            r"$\nu = $" + str(nu_current),
            {"color": "k", "fontsize": fontsize},
        )

    plt.ylabel(r"$u(y)$", fontsize=fontsize)
    plt.xlabel(r"$y$", fontsize=fontsize)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    ax1.set_xlim([-0.05, 0.05])
    ax1.set_ylim([0.0, 0.62])
    plt.savefig("pipe_uProfiles.png", bbox_inches="tight")

    # Distribution of center velocity
    # Predicted result
    num_test = 500
    data_1d_nu_distribution = np.random.normal(NU_MEAN, 0.2 * NU_MEAN, num_test)
    data_2d_xy_test = (
        np.array(np.meshgrid((X_IN - X_OUT) / 2.0, 0, data_1d_nu_distribution))
        .reshape(3, -1)
        .T
    )

    input_dict_test = {
        "x": data_2d_xy_test[:, 0:1],
        "y": data_2d_xy_test[:, 1:2],
        "nu": data_2d_xy_test[:, 2:3],
    }
    output_dict_test = solver.predict(input_dict_test)
    u_max_pred = output_dict_test["u"].numpy()

    # Analytical result, y = 0
    u_max_a = np.zeros([num_test, 1])
    for i in range(num_test):
        u_max_a[i] = (R**2) * dP / (2 * L * data_1d_nu_distribution[i] * RHO)

    # Plot
    plt.figure(2)
    plt.clf()
    ax1 = plt.subplot(111)
    sns.kdeplot(
        u_max_a,
        fill=True,
        color="black",
        label="Analytical",
        linestyle="-",
        linewidth=3,
    )
    sns.kdeplot(
        u_max_pred,
        fill=False,
        color="red",
        label="DNN",
        linestyle="--",
        linewidth=3.5,
    )
    plt.legend(prop={"size": fontsize})
    plt.xlabel(r"$u_c$", fontsize=fontsize)
    plt.ylabel(r"PDF", fontsize=fontsize)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    plt.savefig("pipe_unformUQ.png", bbox_inches="tight")

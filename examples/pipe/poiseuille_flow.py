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

"""
Reference: https://github.com/Jianxun-Wang/LabelFree-DNN-Surrogate
"""

import copy
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import paddle

from ppsci.utils import checker

if not checker.dynamic_import_to_globals("seaborn"):
    raise ModuleNotFoundError("Please install seaborn through pip first.")

import seaborn as sns

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)

    # set output directory
    OUTPUT_DIR = "./output_poiseuille_flow"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

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
    data_2d_xy_shuffle = copy.deepcopy(data_2d_xy)
    np.random.shuffle(data_2d_xy_shuffle)

    input_x = data_2d_xy_shuffle[:, 0].reshape(data_2d_xy_shuffle.shape[0], 1)
    input_y = data_2d_xy_shuffle[:, 1].reshape(data_2d_xy_shuffle.shape[0], 1)
    input_nu = data_2d_xy_shuffle[:, 2].reshape(data_2d_xy_shuffle.shape[0], 1)

    interior_data = {"x": input_x, "y": input_y, "nu": input_nu}
    interior_geom = ppsci.geometry.PointCloud(
        interior={"x": input_x, "y": input_y, "nu": input_nu},
        coord_keys=("x", "y", "nu"),
    )

    # set model
    model_u = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("u",), 3, 50, "swish")
    model_v = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("v",), 3, 50, "swish")
    model_p = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("p",), 3, 50, "swish")

    def input_trans(input):
        x, y = input["x"], input["y"]
        nu = input["nu"]
        b = 2 * np.pi / (X_OUT - X_IN)
        c = np.pi * (X_IN + X_OUT) / (X_IN - X_OUT)
        sin_x = X_IN * paddle.sin(b * x + c)
        cos_x = X_IN * paddle.cos(b * x + c)
        return {"sin(x)": sin_x, "cos(x)": cos_x, "x": x, "y": y, "nu": nu}

    def output_trans_u(input, out):
        return {"u": out["u"] * (R**2 - input["y"] ** 2)}

    def output_trans_v(input, out):
        return {"v": (R**2 - input["y"] ** 2) * out["v"]}

    def output_trans_p(input, out):
        return {
            "p": (
                (P_IN - P_OUT) * (X_OUT - input["x"]) / L
                + (X_IN - input["x"]) * (X_OUT - input["x"]) * out["p"]
            )
        }

    model_u.register_input_transform(input_trans)
    model_v.register_input_transform(input_trans)
    model_p.register_input_transform(input_trans)
    model_u.register_output_transform(output_trans_u)
    model_v.register_output_transform(output_trans_v)
    model_p.register_output_transform(output_trans_p)
    model = ppsci.arch.ModelList((model_u, model_v, model_p))

    # set optimizer
    optimizer = ppsci.optimizer.Adam(5e-3)(model)

    # set euqation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(nu="nu", rho=RHO, dim=2, time=False)
    }

    # set constraint
    BATCH_SIZE = 128
    ITERS_PER_EPOCH = int((N_x * N_y * N_p) / BATCH_SIZE)

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

    # wrap constraints together
    constraint = {pde_constraint.name: pde_constraint}

    EPOCHS = 3000 if not args.epochs else args.epochs

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=False,
        save_freq=10,
        equation=equation,
    )

    solver.train()

    # Cross-section velocity profiles of 4 different viscosity sample
    # Predicted result
    input_dict = {
        "x": data_2d_xy[:, 0:1],
        "y": data_2d_xy[:, 1:2],
        "nu": data_2d_xy[:, 2:3],
    }
    output_dict = solver.predict(input_dict, return_numpy=True)
    u_pred = output_dict["u"].reshape(N_y, N_x, N_p)
    v_pred = output_dict["v"].reshape(N_y, N_x, N_p)
    p_pred = output_dict["p"].reshape(N_y, N_x, N_p)

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
    PLOT_DIR = osp.join(OUTPUT_DIR, "visu")
    os.makedirs(PLOT_DIR, exist_ok=True)
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
        plt.text(
            -0.012,
            ytext[idxP],
            rf"$\nu = $ {data_1d_nu[nu_index[idxP]]}",
            {"color": "k", "fontsize": fontsize},
        )

    plt.ylabel(r"$u(y)$", fontsize=fontsize)
    plt.xlabel(r"$y$", fontsize=fontsize)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    ax1.set_xlim([-0.05, 0.05])
    ax1.set_ylim([0.0, 0.62])
    plt.savefig(osp.join(PLOT_DIR, "pipe_uProfiles.png"), bbox_inches="tight")

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
    output_dict_test = solver.predict(input_dict_test, return_numpy=True)
    u_max_pred = output_dict_test["u"]

    # Analytical result, y = 0
    u_max_a = (R**2) * dP / (2 * L * data_1d_nu_distribution * RHO)

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
    plt.savefig(osp.join(PLOT_DIR, "pipe_unformUQ.png"), bbox_inches="tight")

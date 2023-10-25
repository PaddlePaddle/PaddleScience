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
from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import checker
from ppsci.utils import logger

if not checker.dynamic_import_to_globals("seaborn"):
    raise ModuleNotFoundError("Please install seaborn through pip first.")

import seaborn as sns


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    X_OUT = cfg.X_IN + cfg.L
    Y_START = -cfg.R
    Y_END = Y_START + 2 * cfg.R
    NU_START = cfg.NU_MEAN - cfg.NU_MEAN * cfg.NU_STD  # 0.0001
    NU_END = cfg.NU_MEAN + cfg.NU_MEAN * cfg.NU_STD  # 0.1

    ## prepare data with (?, 2)
    data_1d_x = np.linspace(
        cfg.X_IN, X_OUT, cfg.N_x, endpoint=True, dtype=paddle.get_default_dtype()
    )
    data_1d_y = np.linspace(
        Y_START, Y_END, cfg.N_y, endpoint=True, dtype=paddle.get_default_dtype()
    )
    data_1d_nu = np.linspace(
        NU_START, NU_END, cfg.N_p, endpoint=True, dtype=paddle.get_default_dtype()
    )

    data_2d_xy = (
        np.array(np.meshgrid(data_1d_x, data_1d_y, data_1d_nu)).reshape(3, -1).T
    )
    data_2d_xy_shuffle = copy.deepcopy(data_2d_xy)
    np.random.shuffle(data_2d_xy_shuffle)

    input_x = data_2d_xy_shuffle[:, 0].reshape(data_2d_xy_shuffle.shape[0], 1)
    input_y = data_2d_xy_shuffle[:, 1].reshape(data_2d_xy_shuffle.shape[0], 1)
    input_nu = data_2d_xy_shuffle[:, 2].reshape(data_2d_xy_shuffle.shape[0], 1)

    interior_geom = ppsci.geometry.PointCloud(
        interior={"x": input_x, "y": input_y, "nu": input_nu},
        coord_keys=("x", "y", "nu"),
    )

    # set model
    model_u = ppsci.arch.MLP(**cfg.MODEL.u_net)
    model_v = ppsci.arch.MLP(**cfg.MODEL.v_net)
    model_p = ppsci.arch.MLP(**cfg.MODEL.p_net)

    def input_trans(input):
        x, y = input["x"], input["y"]
        nu = input["nu"]
        b = 2 * np.pi / (X_OUT - cfg.X_IN)
        c = np.pi * (cfg.X_IN + X_OUT) / (cfg.X_IN - X_OUT)
        sin_x = cfg.X_IN * paddle.sin(b * x + c)
        cos_x = cfg.X_IN * paddle.cos(b * x + c)
        return {"sin(x)": sin_x, "cos(x)": cos_x, "x": x, "y": y, "nu": nu}

    def output_trans_u(input, out):
        return {"u": out["u"] * (cfg.R**2 - input["y"] ** 2)}

    def output_trans_v(input, out):
        return {"v": (cfg.R**2 - input["y"] ** 2) * out["v"]}

    def output_trans_p(input, out):
        return {
            "p": (
                (cfg.P_IN - cfg.P_OUT) * (X_OUT - input["x"]) / cfg.L
                + (cfg.X_IN - input["x"]) * (X_OUT - input["x"]) * out["p"]
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
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set euqation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu="nu", rho=cfg.RHO, dim=2, time=False
        )
    }

    # set constraint
    ITERS_PER_EPOCH = int(
        (cfg.N_x * cfg.N_y * cfg.N_p) / cfg.TRAIN.batch_size.pde_constraint
    )

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom=interior_geom,
        dataloader_cfg={
            "dataset": "NamedArrayDataset",
            "num_workers": 1,
            "batch_size": cfg.TRAIN.batch_size.pde_constraint,
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

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=cfg.TRAIN.eval_during_train,
        save_freq=cfg.TRAIN.save_freq,
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
    u_pred = output_dict["u"].reshape(cfg.N_y, cfg.N_x, cfg.N_p)

    # Analytical result, y = data_1d_y
    u_analytical = np.zeros([cfg.N_y, cfg.N_x, cfg.N_p])
    dP = cfg.P_IN - cfg.P_OUT

    for i in range(cfg.N_p):
        uy = (cfg.R**2 - data_1d_y**2) * dP / (2 * cfg.L * data_1d_nu[i] * cfg.RHO)
        u_analytical[:, :, i] = np.tile(uy.reshape([cfg.N_y, 1]), cfg.N_x)

    fontsize = 16
    idx_X = int(round(cfg.N_x / 2))  # pipe velocity section at L/2
    nu_index = [3, 6, 14, 49]  # pick 4 nu samples
    ytext = [0.45, 0.28, 0.1, 0.01]

    # Plot
    PLOT_DIR = osp.join(cfg.output_dir, "visu")
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
    data_1d_nu_distribution = np.random.normal(cfg.NU_MEAN, 0.2 * cfg.NU_MEAN, num_test)
    data_2d_xy_test = (
        np.array(np.meshgrid((cfg.X_IN - X_OUT) / 2.0, 0, data_1d_nu_distribution))
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
    u_max_a = (cfg.R**2) * dP / (2 * cfg.L * data_1d_nu_distribution * cfg.RHO)

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


def evaluate(cfg: DictConfig):
    print("Not supported.")


@hydra.main(version_base=None, config_path="./conf", config_name="poiseuille_flow.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

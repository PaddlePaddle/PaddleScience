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

if not checker.dynamic_import_to_globals("seaborn"):
    raise ModuleNotFoundError("Please install seaborn with `pip install seaborn>=0.13.0`.")  # fmt: skip

import seaborn as sns


def train(cfg: DictConfig):
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


def evaluate(cfg: DictConfig):
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

    # set model
    model_u = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("u",), 3, 50, "swish")
    model_v = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("v",), 3, 50, "swish")
    model_p = ppsci.arch.MLP(("sin(x)", "cos(x)", "y", "nu"), ("p",), 3, 50, "swish")

    class Transform:
        def input_trans(self, input):
            self.input = input
            x, y = input["x"], input["y"]
            nu = input["nu"]
            b = 2 * np.pi / (X_OUT - X_IN)
            c = np.pi * (X_IN + X_OUT) / (X_IN - X_OUT)
            sin_x = X_IN * paddle.sin(b * x + c)
            cos_x = X_IN * paddle.cos(b * x + c)
            return {"sin(x)": sin_x, "cos(x)": cos_x, "y": y, "nu": nu}

        def output_trans_u(self, input, out):
            return {"u": out["u"] * (R**2 - self.input["y"] ** 2)}

        def output_trans_v(self, input, out):
            return {"v": (R**2 - self.input["y"] ** 2) * out["v"]}

        def output_trans_p(self, input, out):
            return {
                "p": (
                    (P_IN - P_OUT) * (X_OUT - self.input["x"]) / L
                    + (X_IN - self.input["x"]) * (X_OUT - self.input["x"]) * out["p"]
                )
            }

    transform = Transform()
    model_u.register_input_transform(transform.input_trans)
    model_v.register_input_transform(transform.input_trans)
    model_p.register_input_transform(transform.input_trans)
    model_u.register_output_transform(transform.output_trans_u)
    model_v.register_output_transform(transform.output_trans_v)
    model_p.register_output_transform(transform.output_trans_p)
    model = ppsci.arch.ModelList((model_u, model_v, model_p))

    # Validator vel
    input_dict = {
        "x": data_2d_xy[:, 0:1],
        "y": data_2d_xy[:, 1:2],
        "nu": data_2d_xy[:, 2:3],
    }
    u_analytical = np.zeros([N_y, N_x, N_p])
    dP = P_IN - P_OUT

    for i in range(N_p):
        uy = (R**2 - data_1d_y**2) * dP / (2 * L * data_1d_nu[i] * RHO)
        u_analytical[:, :, i] = np.tile(uy.reshape([N_y, 1]), N_x)

    label_dict = {"u": np.ones_like(input_dict["x"])}
    weight_dict = {"u": np.ones_like(input_dict["x"])}

    # Validator KL
    num_test = 500
    data_1d_nu_distribution = np.random.normal(NU_MEAN, 0.2 * NU_MEAN, num_test)
    data_2d_xy_test = (
        np.array(
            np.meshgrid((X_IN - X_OUT) / 2.0, 0, data_1d_nu_distribution), np.float32
        )
        .reshape(3, -1)
        .T
    )
    input_dict_KL = {
        "x": data_2d_xy_test[:, 0:1],
        "y": data_2d_xy_test[:, 1:2],
        "nu": data_2d_xy_test[:, 2:3],
    }
    u_max_a = (R**2) * dP / (2 * L * data_1d_nu_distribution * RHO)
    label_dict_KL = {"u": np.ones_like(input_dict_KL["x"])}
    weight_dict_KL = {"u": np.ones_like(input_dict_KL["x"])}

    class Cross_section_velocity_profile_metric(ppsci.metric.base.Metric):
        def __init__(self, keep_batch: bool = False):
            super().__init__(keep_batch)

        @paddle.no_grad()
        def forward(self, output_dict, label_dict):
            u_pred = output_dict["u"].numpy().reshape(N_y, N_x, N_p)
            metric_dict = {}
            for nu in range(N_p):
                err = (
                    u_analytical[:, int(round(N_x / 2)), nu]
                    - u_pred[:, int(round(N_x / 2)), nu]
                )
                metric_dict[f"nu = {data_1d_nu[nu]:.2g}"] = np.abs(err).sum()
            return metric_dict

    # Kullback-Leibler Divergence
    class KL_divergence(ppsci.metric.base.Metric):
        def __init__(self, keep_batch: bool = False):
            super().__init__(keep_batch)

        @paddle.no_grad()
        def forward(self, output_dict, label_dict):
            u_max_pred = output_dict["u"].numpy().flatten()
            import scipy

            print(f"KL = {scipy.stats.entropy(u_max_a, u_max_pred)}")
            return {"KL divergence": scipy.stats.entropy(u_max_a, u_max_pred)}

    dataset_vel = {
        "name": "NamedArrayDataset",
        "input": input_dict,
        "label": label_dict,
        "weight": weight_dict,
    }
    dataset_kl = {
        "name": "NamedArrayDataset",
        "input": input_dict_KL,
        "label": label_dict_KL,
        "weight": weight_dict_KL,
    }
    eval_cfg = {
        "sampler": {
            "name": "BatchSampler",
            "shuffle": False,
            "drop_last": False,
        },
        "batch_size": 2000,
    }
    eval_cfg["dataset"] = dataset_vel
    velocity_validator = ppsci.validate.SupervisedValidator(
        eval_cfg,
        ppsci.loss.MSELoss("mean"),
        {"u": lambda out: out["u"]},
        {"Cross_section_velocity_profile_MAE": Cross_section_velocity_profile_metric()},
        name="Cross_section_velocity_profile_MAE",
    )
    eval_cfg["dataset"] = dataset_kl
    kl_validator = ppsci.validate.SupervisedValidator(
        eval_cfg,
        ppsci.loss.MSELoss("mean"),
        {"u": lambda out: out["u"]},
        {"KL_divergence": KL_divergence()},
        name="KL_divergence",
    )
    validator = {
        velocity_validator.name: velocity_validator,
        kl_validator.name: kl_validator,
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    solver.eval()

    output_dict = solver.predict(input_dict, return_numpy=True)
    u_pred = output_dict["u"].reshape(N_y, N_x, N_p)
    fontsize = 16
    idx_X = int(round(N_x / 2))  # pipe velocity section at L/2
    nu_index = [3, 6, 9, 12, 14, 20, 49]  # pick 7 nu samples
    ytext = [0.55, 0.5, 0.4, 0.28, 0.1, 0.05, 0.001]  # text y position

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
            rf"$\nu = $ {data_1d_nu[nu_index[idxP]]:.2g}",
            {"color": "k", "fontsize": fontsize - 4},
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


def export(cfg: DictConfig):
    from paddle.static import InputSpec

    model_u = ppsci.arch.MLP(**cfg.MODEL.u_net)
    model_v = ppsci.arch.MLP(**cfg.MODEL.v_net)
    model_p = ppsci.arch.MLP(**cfg.MODEL.p_net)
    X_OUT = cfg.X_IN + cfg.L

    class Transform:
        def input_trans(self, input):
            self.input = input
            x, y = input["x"], input["y"]
            nu = input["nu"]
            b = 2 * np.pi / (X_OUT - cfg.X_IN)
            c = np.pi * (cfg.X_IN + X_OUT) / (cfg.X_IN - X_OUT)
            sin_x = cfg.X_IN * paddle.sin(b * x + c)
            cos_x = cfg.X_IN * paddle.cos(b * x + c)
            return {"sin(x)": sin_x, "cos(x)": cos_x, "y": y, "nu": nu}

        def output_trans_u(self, input, out):
            return {"u": out["u"] * (cfg.R**2 - self.input["y"] ** 2)}

        def output_trans_v(self, input, out):
            return {"v": (cfg.R**2 - self.input["y"] ** 2) * out["v"]}

        def output_trans_p(self, input, out):
            return {
                "p": (
                    (cfg.P_IN - cfg.P_OUT) * (X_OUT - self.input["x"]) / cfg.L
                    + (cfg.X_IN - self.input["x"])
                    * (X_OUT - self.input["x"])
                    * out["p"]
                )
            }

    transform = Transform()
    model_u.register_input_transform(transform.input_trans)
    model_v.register_input_transform(transform.input_trans)
    model_p.register_input_transform(transform.input_trans)
    model_u.register_output_transform(transform.output_trans_u)
    model_v.register_output_transform(transform.output_trans_v)
    model_p.register_output_transform(transform.output_trans_p)
    model = ppsci.arch.ModelList((model_u, model_v, model_p))

    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    input_keys = ["x", "y", "nu"]
    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
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

    # Initialize your custom predictor
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # Prepare input data
    input_dict = {
        "x": data_2d_xy[:, 0:1],
        "y": data_2d_xy[:, 1:2],
        "nu": data_2d_xy[:, 2:3],
    }

    u_analytical = np.zeros([N_y, N_x, N_p])
    dP = P_IN - P_OUT

    for i in range(N_p):
        uy = (R**2 - data_1d_y**2) * dP / (2 * L * data_1d_nu[i] * RHO)
        u_analytical[:, :, i] = np.tile(uy.reshape([N_y, 1]), N_x)

    # Validator KL
    num_test = 500
    data_1d_nu_distribution = np.random.normal(NU_MEAN, 0.2 * NU_MEAN, num_test)
    data_2d_xy_test = (
        np.array(
            np.meshgrid((X_IN - X_OUT) / 2.0, 0, data_1d_nu_distribution), np.float32
        )
        .reshape(3, -1)
        .T
    )

    # Perform inference
    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }
    # Process and reshape output as needed
    u_pred = output_dict["u"].reshape(N_y, N_x, N_p)
    fontsize = 16
    idx_X = int(round(N_x / 2))  # pipe velocity section at L/2
    nu_index = [3, 6, 9, 12, 14, 20, 49]  # pick 7 nu samples
    ytext = [0.55, 0.5, 0.4, 0.28, 0.1, 0.05, 0.001]  # text y position

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
            rf"$\nu = $ {data_1d_nu[nu_index[idxP]]:.2g}",
            {"color": "k", "fontsize": fontsize - 4},
        )

    plt.ylabel(r"$u(y)$", fontsize=fontsize)
    plt.xlabel(r"$y$", fontsize=fontsize)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    ax1.set_xlim([-0.05, 0.05])
    ax1.set_ylim([0.0, 0.62])
    plt.savefig(osp.join(PLOT_DIR, "pipe_uProfiles.png"), bbox_inches="tight")

    # Distribution of center velocity
    num_test = 500
    data_1d_nu_distribution = np.random.normal(NU_MEAN, 0.2 * NU_MEAN, num_test)
    data_2d_xy_test = (
        np.array(
            np.meshgrid((X_IN - X_OUT) / 2.0, 0, data_1d_nu_distribution), np.float32
        )
        .reshape(3, -1)
        .T
    )
    # Predicted result
    input_dict_test = {
        "x": data_2d_xy_test[:, 0:1],
        "y": data_2d_xy_test[:, 1:2],
        "nu": data_2d_xy_test[:, 2:3],
    }
    output_dict_test = predictor.predict(input_dict_test, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_dict_test = {
        store_key: output_dict_test[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict_test.keys())
    }
    u_max_pred = output_dict_test["u"]
    u_max_a = (R**2) * dP / (2 * L * data_1d_nu_distribution * RHO)

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


@hydra.main(version_base=None, config_path="./conf", config_name="poiseuille_flow.yaml")
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

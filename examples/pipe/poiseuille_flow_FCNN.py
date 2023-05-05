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


class NavierStokes(base.PDE):
    """Class for navier-stokes equation. [nu] as self-variable

    Args:
        rho (float): Density.
        dim (int): Dimension of equation.
        time (bool): Whether the euqation is time-dependent.
    """

    def __init__(self, rho: float, dim: int, time: bool):
        super().__init__()
        self.rho = rho
        self.dim = dim
        self.time = time

        def continuity_compute_func(out):
            x, y, nu = out["x"], out["y"], out["nu"]
            u, v = out["u"], out["v"]
            continuity = jacobian(u, x) + jacobian(v, y)

            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                continuity += jacobian(w, z)
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def momentum_x_compute_func(out):
            x, y, nu = out["x"], out["y"], out["nu"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_x = (
                u * jacobian(u, x)
                + v * jacobian(u, y)
                - nu / rho * hessian(u, x)
                - nu / rho * hessian(u, y)
                + 1 / rho * jacobian(p, x)
            )
            if self.time:
                t = out["t"]
                momentum_x += jacobian(u, t)
            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                momentum_x += w * jacobian(u, z)
                momentum_x -= nu / rho * hessian(u, z)
            return momentum_x

        self.add_equation("momentum_x", momentum_x_compute_func)

        def momentum_y_compute_func(out):
            x, y, nu = out["x"], out["y"], out["nu"]
            u, v, p = out["u"], out["v"], out["p"]
            momentum_y = (
                u * jacobian(v, x)
                + v * jacobian(v, y)
                - nu / rho * hessian(v, x)
                - nu / rho * hessian(v, y)
                + 1 / rho * jacobian(p, y)
            )
            if self.time:
                t = out["t"]
                momentum_y += jacobian(v, t)
            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                momentum_y += w * jacobian(v, z)
                momentum_y -= nu / rho * hessian(v, z)
            return momentum_y

        self.add_equation("momentum_y", momentum_y_compute_func)

        if self.dim == 3:

            def momentum_z_compute_func(out):
                x, y, nu = out["x"], out["y"], out["nu"]
                u, v, w, p = out["u"], out["v"], out["w"], out["p"]
                momentum_z = (
                    u * jacobian(w, x)
                    + v * jacobian(w, y)
                    + w * jacobian(w, z)
                    - nu / rho * hessian(w, x)
                    - nu / rho * hessian(w, y)
                    - nu / rho * hessian(w, z)
                    + 1 / rho * jacobian(p, z)
                )
                if self.time:
                    t = out["t"]
                    momentum_z += jacobian(w, t)
                return momentum_z

            self.add_equation("momentum_z", momentum_z_compute_func)


def predict(
    input_dict,
    solver,
):
    for key, val in input_dict.items():
        input_dict[key] = paddle.to_tensor(val, dtype="float32")
    evaluator = ppsci.utils.expression.ExpressionSolver(
        input_dict.keys(), ["u", "v", "p"], solver.model
    )
    output_expr_dict = {
        "u": lambda d: d["u"],
        "v": lambda d: d["v"],
        "p": lambda d: d["p"],
    }
    for output_key, output_expr in output_expr_dict.items():
        evaluator.add_target_expr(output_expr, output_key)
    output_dict = evaluator(input_dict)
    return output_dict


if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)

    os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/pipe")
    # set output directory
    output_dir = "./output"
    dir = "./data/net_params"

    core.set_prim_eager_enabled(True)

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

    HIDDEN_SIZE = 50
    LAYER_NUMBER = 4 - 1  # last fc
    LOG_FREQ = 1
    EVAL_FREQ = 100  # display step
    VISU_FREQ = 100  # visulize step

    X_IN = 0
    X_OUT = X_IN + L
    Y_START = -R
    Y_END = Y_START + 2 * R
    NU_START = NU_MEAN - NU_MEAN * NU_STD  # 0.0001
    NU_END = NU_MEAN + NU_MEAN * NU_STD  # 0.1

    ## prepare data with (?, 2)
    data_1d_x = np.linspace(X_IN, X_OUT, N_x, endpoint=True)
    data_1d_y = np.linspace(Y_START, Y_END, N_y, endpoint=True)
    data_1d_nu = np.linspace(NU_START, NU_END, N_p, endpoint=True)

    data_2d_xy = (
        np.array(np.meshgrid(data_1d_x, data_1d_y, data_1d_nu)).reshape(3, -1).T
    )
    data_2d_xy_old = copy.deepcopy(data_2d_xy)
    np.random.shuffle(data_2d_xy)

    input_x = data_2d_xy[:, 0].reshape(data_2d_xy.shape[0], 1).astype("float32")
    input_y = data_2d_xy[:, 1].reshape(data_2d_xy.shape[0], 1).astype("float32")
    input_nu = data_2d_xy[:, 2].reshape(data_2d_xy.shape[0], 1).astype("float32")

    interior_data = {"x": input_x, "y": input_y, "nu": input_nu}
    interior_geom = ppsci.geometry.PointCloud(
        coord_dict={"x": input_x, "y": input_y},
        extra_data={"nu": input_nu},
        data_key=["x", "y", "nu"],
    )

    # set model
    model_u = ppsci.arch.MLP(
        ["sin(x)", "cos(x)", "y", "nu"],
        ["u"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish_beta",
        False,
        False,
        np.load(dir + f"/weight_u_epoch_1.npz"),
        np.load(dir + f"/bias_u_epoch_1.npz"),
    )

    model_v = ppsci.arch.MLP(
        ["sin(x)", "cos(x)", "y", "nu"],
        ["v"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish_beta",
        False,
        False,
        np.load(dir + f"/weight_v_epoch_1.npz"),
        np.load(dir + f"/bias_v_epoch_1.npz"),
    )

    model_p = ppsci.arch.MLP(
        ["sin(x)", "cos(x)", "y", "nu"],
        ["p"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish_beta",
        False,
        False,
        np.load(dir + f"/weight_p_epoch_1.npz"),
        np.load(dir + f"/bias_p_epoch_1.npz"),
    )

    def output_transform(out, input):
        new_out = {}
        x, y = input["x"], input["y"]

        if next(iter(out.keys())) == "u":
            u = out["u"]
            # The no-slip condition of velocity on the wall
            new_out["u"] = u * (R**2 - y**2)
        elif next(iter(out.keys())) == "v":
            v = out["v"]
            # The no-slip condition of velocity on the wall
            new_out["v"] = (R**2 - y**2) * v
        elif next(iter(out.keys())) == "p":
            p = out["p"]
            # The pressure inlet [p_in = 0.1] and outlet [p_out = 0]
            new_out["p"] = (
                (X_IN - x) * 0
                + (P_IN - P_OUT) * (X_OUT - x) / L
                + 0 * y
                + (X_IN - x) * (X_OUT - x) * p
            )
        else:
            raise NotImplementedError(f"{out.keys()} are outputs to be implemented")

        return new_out

    def input_transform(input):
        x, y = input["x"], input["y"]
        nu = input["nu"]
        b = 2 * np.pi / (X_OUT - X_IN)
        c = np.pi * (X_IN + X_OUT) / (X_IN - X_OUT)
        sin_x = X_IN * paddle.sin(b * x + c)
        cos_x = X_IN * paddle.cos(b * x + c)
        return {"sin(x)": sin_x, "cos(x)": cos_x, "y": y, "nu": nu}

    model_u.register_input_transform(input_transform)
    model_v.register_input_transform(input_transform)
    model_p.register_input_transform(input_transform)
    model_u.register_output_transform(output_transform)
    model_v.register_output_transform(output_transform)
    model_p.register_output_transform(output_transform)
    model = ppsci.arch.ModelList([model_u, model_v, model_p])

    # set optimizer
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE)([model])

    # set euqation
    equation = {"NavierStokes": NavierStokes(RHO, 2, False)}

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
        weight_dict={"u": 1, "v": 1, "p": 1},
        name="EQ",
    )

    visualizer = {
        "visulzie_u": ppsci.visualize.VisualizerVtu(
            interior_data,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            VISU_FREQ,
            "result_u",
        )
    }

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
        log_freq=LOG_FREQ,
        equation=equation,
        checkpoint_path="/workspace/wangguan/PaddleScience_Surrogate/examples/pipe/output/checkpoints/epoch_3000",
    )

    # Cross-section velocity profiles of 4 different viscosity sample
    # Predicted result
    input_dict = {
        "x": data_2d_xy_old[:, 0:1],
        "y": data_2d_xy_old[:, 1:2],
        "nu": data_2d_xy_old[:, 2:3],
    }
    output_dict = predict(input_dict, solver)
    u_pred = output_dict["u"].numpy().reshape(N_y, N_x, N_p)
    v_pred = output_dict["v"].numpy().reshape(N_y, N_x, N_p)
    p_pred = output_dict["p"].numpy().reshape(N_y, N_x, N_p)

    # Analytical result, y = data_1d_y
    uSolaM = np.zeros([N_y, N_x, N_p])
    dP = P_IN - P_OUT

    for i in range(N_p):
        uy = (R**2 - data_1d_y**2) * dP / (2 * L * data_1d_nu[i] * RHO)
        uSolaM[:, :, i] = np.tile(uy.reshape([N_y, 1]), N_x)

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
            uSolaM[:, idx_X, nu_index[idxP]],
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
    N_pTest = 500
    data_1d_nuDist = np.random.normal(NU_MEAN, 0.2 * NU_MEAN, N_pTest)
    data_2d_xy_test = (
        np.array(np.meshgrid((X_IN - X_OUT) / 2.0, 0, data_1d_nuDist)).reshape(3, -1).T
    )

    input_dict_test = {
        "x": data_2d_xy_test[:, 0:1],
        "y": data_2d_xy_test[:, 1:2],
        "nu": data_2d_xy_test[:, 2:3],
    }
    output_dict_test = predict(input_dict_test, solver)
    uMax_pred = output_dict_test["u"].numpy()

    # Analytical result, y = 0
    uMax_a = np.zeros([N_pTest, 1])
    for i in range(N_pTest):
        uMax_a[i] = (R**2) * dP / (2 * L * data_1d_nuDist[i] * RHO)

    # Plot
    plt.figure(2)
    plt.clf()
    ax1 = plt.subplot(111)
    sns.kdeplot(
        uMax_a, fill=True, color="black", label="Analytical", linestyle="-", linewidth=3
    )
    sns.kdeplot(
        uMax_pred,
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

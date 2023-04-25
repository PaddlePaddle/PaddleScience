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

import os.path as osp

import numpy as np
import paddle

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


if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    import os

    os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/pipe")
    # set output directory
    output_dir = "./output_pipe"
    initial_bias_u = np.load("./data/init_net_params/initial_bias_u.npz")
    initial_bias_v = np.load("./data/init_net_params/initial_bias_v.npz")
    initial_bias_p = np.load("./data/init_net_params/initial_bias_p.npz")
    initial_weight_u = np.load("./data/init_net_params/initial_weight_u.npz")
    initial_weight_v = np.load("./data/init_net_params/initial_weight_v.npz")
    initial_weight_p = np.load("./data/init_net_params/initial_weight_p.npz")

    from paddle.fluid import core

    core.set_prim_eager_enabled(True)

    # initialize logger
    ppsci.utils.logger.init_logger("ppsci", f"{output_dir}/train.log", "info")

    re = 200.0  # Reynolds number re = U(2R)/nu
    nuMean = 0.001
    nuStd = 0.9
    L = 1.0  # length of pipe
    R = 0.05  # radius of pipe
    RHO = 1  # density
    P_OUT = 0  # pressure at the outlet of pipe
    P_IN = 0.1  # pressure at the inlet of pipe
    periodicBC = True  # or false

    eps = 1e-4
    coef_reg = 1e-5

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

    yStart = -R
    yEnd = yStart + 2 * R

    nuStart = nuMean - nuMean * nuStd  # 0.0001
    nuEnd = nuMean + nuMean * nuStd  # 0.1

    ## prepare data with (?, 2)
    data_1d_x = np.linspace(X_IN, X_OUT, N_x, endpoint=True)
    data_1d_y = np.linspace(yStart, yEnd, N_y, endpoint=True)
    data_1d_nu = np.linspace(nuStart, nuEnd, N_p, endpoint=True)

    print("train_nu is", data_1d_nu)

    data_2d_xy_before = np.array(np.meshgrid(data_1d_x, data_1d_y, data_1d_nu))
    data_2d_xy_before_reshape = data_2d_xy_before.reshape(3, -1)
    data_2d_xy = data_2d_xy_before_reshape.T
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
    np.savez("./data/input/input_x_y_nu", x=input_x, y=input_y, nu=input_nu)
    # set model
    model_u = ppsci.arch.MLP(
        ["sin(x)", "cos(x)", "y", "nu"],
        ["u"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish",
        False,
        False,
        initial_weight_u,
        initial_bias_u,
    )

    model_v = ppsci.arch.MLP(
        ["sin(x)", "cos(x)", "y", "nu"],
        ["v"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish",
        False,
        False,
        initial_weight_v,
        initial_bias_v,
    )

    model_p = ppsci.arch.MLP(
        ["sin(x)", "cos(x)", "y", "nu"],
        ["p"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish",
        False,
        False,
        initial_weight_p,
        initial_bias_p,
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
            # new_out["p"] =  (x - X_IN) / L * P_IN + (x - X_OUT) / L * P_OUT + (x - X_IN) * (X_OUT - x) * p
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
        if periodicBC == True:
            x, y = input["x"], input["y"]
            print(x.stop_gradient, y.stop_gradient)
            nu = input["nu"]
            b = 2 * np.pi / (X_OUT - X_IN)
            c = np.pi * (X_IN + X_OUT) / (X_IN - X_OUT)
            sin_x = X_IN * paddle.sin(b * x + c)
            cos_x = X_IN * paddle.cos(b * x + c)
            return {"sin(x)": sin_x, "cos(x)": cos_x, "y": y, "nu": nu}
        else:
            pass

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
        eval_during_train=True,
        log_freq=LOG_FREQ,
        eval_freq=EVAL_FREQ,
        equation=equation,
        visualizer=visualizer,
    )

    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()

    # visualize prediction after finished training
    solver.visualize()

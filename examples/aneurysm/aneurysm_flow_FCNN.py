import math
import os

import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.fluid import core

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian


def ThreeD_mesh(x_2d, tmp_1d):
    tmp_3d = np.expand_dims(np.tile(tmp_1d, len(x_2d)), 1).astype("float")
    x = []
    for x0 in x_2d:
        tmpx = np.tile(x0, len(tmp_1d))
        x.append(tmpx)
    x = np.reshape(x, (len(tmp_3d), 1))
    return x, tmp_3d


class NavierStokes(ppsci.equation.pde.base.PDE):
    """Class for navier-stokes equation.

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
            x, y = out["x"], out["y"]
            u, v = out["u"], out["v"]
            continuity = jacobian(u, x) + jacobian(v, y)

            if self.dim == 3:
                z = out["z"]
                w = out["w"]
                continuity += jacobian(w, z)
            return continuity

        self.add_equation("continuity", continuity_compute_func)

        def momentum_x_compute_func(out):
            x, y = out["x"], out["y"]
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
            x, y = out["x"], out["y"]
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
                x, y = out["x"], out["y"]
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
    os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm")
    output_dir = "./output_0504_debug"
    ppsci.utils.misc.set_random_seed(42)

    # initialize logger
    ppsci.utils.logger.init_logger("ppsci", f"{output_dir}/train.log", "info")
    core.set_prim_eager_enabled(True)

    EPOCHS = 500
    L = 1
    X_IN = 0
    X_OUT = X_IN + L
    P_OUT = 0  # pressure at the outlet of pipe
    P_IN = 0.1  # pressure at the inlet of pipe
    rInlet = 0.05
    nPt = 100
    unique_x = np.linspace(X_IN, X_OUT, nPt)
    mu = 0.5 * (X_OUT - X_IN)
    N_y = 20
    x_2d = np.tile(unique_x, N_y)
    x_2d = np.reshape(x_2d, (len(x_2d), 1))
    nu = 1e-3
    sigma = 0.1
    scaleStart = -0.02
    scaleEnd = 0
    Ng = 50
    scale_1d = np.linspace(scaleStart, scaleEnd, Ng, endpoint=True)
    x, scale = ThreeD_mesh(x_2d, scale_1d)

    # axisymetric boundary
    R = (
        scale
        * 1
        / math.sqrt(2 * np.pi * sigma**2)
        * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )

    # Generate stenosis
    yUp = (rInlet - R) * np.ones_like(x)
    yDown = (-rInlet + R) * np.ones_like(x)
    idx = np.where(scale == scaleStart)
    plt.figure()
    plt.scatter(x[idx], yUp[idx])
    plt.scatter(x[idx], yDown[idx])
    plt.axis("equal")
    plt.show()
    plt.savefig("idealized_stenotid_vessel", bbox_inches="tight")

    y = np.zeros([len(x), 1])
    for x0 in unique_x:
        index = np.where(x[:, 0] == x0)[0]
        Rsec = max(yUp[index])
        tmpy = np.linspace(-Rsec, Rsec, len(index)).reshape(len(index), -1)
        y[index] = tmpy

    RHO = 1
    BATCH_SIZE = 50
    LEARNING_RATE = 1e-3
    LAYER_NUMBER = 4 - 1
    HIDDEN_SIZE = 20

    index = [i for i in range(x.shape[0])]
    res = list(zip(x, y, scale))
    np.random.shuffle(res)
    x, y, scale = zip(*res)
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    scale = np.array(scale).astype(float)

    interior_geom = ppsci.geometry.PointCloud(
        coord_dict={"x": x, "y": y},
        extra_data={"scale": scale},
        data_key=["x", "y", "scale"],
    )

    model_2 = ppsci.arch.MLP(
        ["x", "y", "scale"],
        ["u"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish",
        False,
        False,
        np.load(f"data/net2_params/weight_epoch_0.npz"),
        np.load(f"data/net2_params/bias_epoch_0.npz"),
    )

    model_3 = ppsci.arch.MLP(
        ["x", "y", "scale"],
        ["v"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish",
        False,
        False,
        np.load(f"data/net3_params/weight_epoch_0.npz"),
        np.load(f"data/net3_params/bias_epoch_0.npz"),
    )

    model_4 = ppsci.arch.MLP(
        ["x", "y", "scale"],
        ["p"],
        LAYER_NUMBER,
        HIDDEN_SIZE,
        "swish",
        False,
        False,
        np.load(f"data/net4_params/weight_epoch_0.npz"),
        np.load(f"data/net4_params/bias_epoch_0.npz"),
    )

    class Output_transform:
        def __init__(self) -> None:
            pass

        def __call__(self, out, input):
            new_out = {}
            x, y, scale = input["x"], input["y"], input["scale"]
            # axisymetric boundary
            if next(iter(out.keys())) == "u":
                R = (
                    scale
                    * 1
                    / np.sqrt(2 * np.pi * sigma**2)
                    * paddle.exp(-((x - mu) ** 2) / (2 * sigma**2))
                )
                self.h = rInlet - R
                u = out["u"]
                # The no-slip condition of velocity on the wall
                new_out["u"] = u * (self.h**2 - y**2)
            elif next(iter(out.keys())) == "v":
                v = out["v"]
                # The no-slip condition of velocity on the wall
                new_out["v"] = (self.h**2 - y**2) * v
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

    shared_transform = Output_transform()
    model_2.register_output_transform(shared_transform)
    model_3.register_output_transform(shared_transform)
    model_4.register_output_transform(shared_transform)
    model = ppsci.arch.ModelList([model_2, model_3, model_4])

    optimizer2 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=10**-15
    )([model])

    equation = {"NavierStokes": NavierStokes(RHO, 2, False)}

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom=interior_geom,
        dataloader_cfg={
            "dataset": "NamedArrayDataset",
            "num_workers": 1,
            "batch_size": BATCH_SIZE,
            "iters_per_epoch": int(x.shape[0] / BATCH_SIZE),
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

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        {pde_constraint.name: pde_constraint},
        output_dir,
        optimizer2,
        epochs=EPOCHS,
        iters_per_epoch=int(x.shape[0] / BATCH_SIZE),
        eval_during_train=False,
        save_freq=10,
        log_freq=1,
        equation=equation,
        checkpoint_path="/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm/output_0504/checkpoints/epoch_280",
    )

    # solver.train()
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

    def single_test(x, y, scale, caseIdx):
        xt = paddle.to_tensor(x, dtype="float32")
        yt = paddle.to_tensor(y, dtype="float32")
        scalet = scale * paddle.ones_like(xt)
        net_in = {"x": xt, "y": yt, "scale": scalet}
        output_dict = predict(net_in, solver)
        np.savez(
            "./" + str(int(caseIdx)) + "ML_WallStress_uvp",
            x_center=x,
            y_center=y,
            u_center=output_dict["u"],
            v_center=output_dict["v"],
            p_center=output_dict["p"],
        )
        return output_dict

    scale_test = np.load("aneurysm_scale0005to002_eval0to002mean001_3sigma.npz")[
        "scale"
    ]
    os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm")
    caseCount = [1.0, 151.0, 486.0]
    W_ctl = np.zeros([len(scale_test), 1])
    W_ctl_Ml = np.zeros([len(scale_test), 1])
    plot_x = 0.8
    plot_y = 0.06
    fontsize = 14
    axis_limit = [0, 1, -0.15, 0.15]
    path = "./data/cases/"
    for caseIdx in caseCount:
        scale = scale_test[int(caseIdx - 1)]
        Data_CFD = np.load(path + str(caseIdx) + "CFD_contour.npz")
        Data_NN = np.load(path + str(caseIdx) + "NN_contour.npz")
        x = Data_CFD["x"]
        y = Data_CFD["y"]
        U_CFD = Data_CFD["U"]
        U = Data_NN["U"]
        n = len(x)
        output_dict = single_test(
            x.reshape(n, 1), y.reshape(n, 1), np.ones((n, 1)) * scale, caseIdx
        )
        u, v, p = output_dict["u"], output_dict["v"], output_dict["p"]
        w = np.zeros_like(u)
        U = np.concatenate([u, v, w], axis=1)

        # velocity U
        plt.figure()
        plt.subplot(212)
        plt.scatter(x, y, c=U[:, 0], vmin=min(U_CFD[:, 0]), vmax=max(U_CFD[:, 0]))
        plt.text(plot_x, plot_y, r"DNN", {"color": "b", "fontsize": fontsize})
        plt.axis(axis_limit)
        plt.colorbar()
        plt.subplot(211)
        plt.scatter(x, y, c=U_CFD[:, 0], vmin=min(U_CFD[:, 0]), vmax=max(U_CFD[:, 0]))
        plt.colorbar()
        plt.text(plot_x, plot_y, r"CFD", {"color": "b", "fontsize": fontsize})
        plt.axis(axis_limit)
        plt.savefig(
            "plot/" + str(int(caseIdx)) + "_scale_" + str(scale) + "_uContour_test.png",
            bbox_inches="tight",
        )

        # velocity V
        plt.figure()
        plt.subplot(212)
        plt.scatter(x, y, c=U[:, 1], vmin=min(U_CFD[:, 1]), vmax=max(U_CFD[:, 1]))
        plt.text(plot_x, plot_y, r"DNN", {"color": "b", "fontsize": fontsize})
        plt.axis(axis_limit)
        plt.colorbar()
        plt.subplot(211)
        plt.scatter(x, y, c=U_CFD[:, 1], vmin=min(U_CFD[:, 1]), vmax=max(U_CFD[:, 1]))
        plt.colorbar()
        plt.text(plot_x, plot_y, r"CFD", {"color": "b", "fontsize": fontsize})
        plt.axis(axis_limit)
        plt.savefig(
            "plot/" + str(int(caseIdx)) + "_scale_" + str(scale) + "_vContour_test.png",
            bbox_inches="tight",
        )
        plt.close("all")

        # wall shear stress
        Data_CFD_wss = np.load(path + str(caseIdx) + "CFD_wss.npz")
        unique_x = Data_CFD_wss["x"]
        wall_shear_mag_up = Data_CFD_wss["wss"]
        Data_NN_wss = np.load(path + str(caseIdx) + "NN_wss.npz")
        NNwall_shear_mag_up = Data_NN_wss["wss"]

        plt.figure()
        plt.plot(
            unique_x,
            wall_shear_mag_up,
            label="CFD",
            color="darkblue",
            linestyle="-",
            lw=3.0,
            alpha=1.0,
        )
        plt.plot(
            unique_x,
            NNwall_shear_mag_up,
            label="DNN",
            color="red",
            linestyle="--",
            dashes=(5, 5),
            lw=2.0,
            alpha=1.0,
        )
        plt.xlabel(r"x", fontsize=16)
        plt.ylabel(r"$\tau_{c}$", fontsize=16)
        plt.legend(prop={"size": 16})
        plt.savefig(
            "plot/" + str(int(caseIdx)) + "_nu_" + str(nu) + "_wallshear_test.png",
            bbox_inches="tight",
        )
        plt.close("all")

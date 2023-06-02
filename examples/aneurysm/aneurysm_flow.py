import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.fluid import core

import ppsci
from ppsci.utils import misc

if __name__ == "__main__":
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)

    # set output directory
    OUTPUT_DIR = "./output_0601"
    PLOT_DIR = osp.join(OUTPUT_DIR, "visu")
    os.makedirs(PLOT_DIR, exist_ok=True)
    # initialize logger
    ppsci.utils.logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    core.set_prim_eager_enabled(True)

    # Hyper parameters
    EPOCHS = 500
    BATCH_SIZE = 50
    LEARNING_RATE = 1e-3

    # Physic properties
    P_OUT = 0  # pressure at the outlet of pipe
    P_IN = 0.1  # pressure at the inlet of pipe
    NU = 1e-3
    RHO = 1

    # Geometry
    L = 1
    X_IN = 0
    X_OUT = X_IN + L
    R_INLET = 0.05
    unique_x = np.linspace(X_IN, X_OUT, 100)
    mu = 0.5 * (X_OUT - X_IN)
    N_Y = 20
    x_inital = np.linspace(X_IN, X_OUT, 100, dtype=paddle.get_default_dtype()).reshape(
        100, 1
    )
    x_20_copy = np.tile(x_inital, (20, 1))  # duplicate 20 times of x for dataloader
    SIGMA = 0.1
    SCALE_START = -0.02
    SCALE_END = 0
    scale_initial = np.linspace(
        SCALE_START, SCALE_END, 50, endpoint=True, dtype=paddle.get_default_dtype()
    ).reshape(50, 1)
    scale = np.tile(scale_initial, (len(x_20_copy), 1))
    x = np.array([np.tile(val, len(scale_initial)) for val in x_20_copy]).reshape(
        len(scale), 1
    )

    # Axisymetric boundary
    R = (
        scale
        * 1
        / math.sqrt(2 * np.pi * SIGMA**2)
        * np.exp(-((x - mu) ** 2) / (2 * SIGMA**2))
    )

    # Visualize stenosis(scale == 0.2)
    y_up = (R_INLET - R) * np.ones_like(x)
    y_down = (-R_INLET + R) * np.ones_like(x)
    idx = np.where(scale == 0)  # plot vessel which scale is 0.2 by finding its indexs
    plt.figure()
    plt.scatter(x[idx], y_up[idx])
    plt.scatter(x[idx], y_down[idx])
    plt.axis("equal")
    plt.savefig(osp.join(PLOT_DIR, "idealized_stenotid_vessel"), bbox_inches="tight")

    # Points and shuffle(for alignment)
    y = np.zeros([len(x), 1], dtype=paddle.get_default_dtype())
    for x0 in x_inital:
        index = np.where(x[:, 0] == x0)[0]
        # y is linear to scale, so we place linespace to get 1000 x, it coressponds to vessels
        y[index] = np.linspace(-max(y_up[index]), max(y_up[index]), len(index)).reshape(
            len(index), -1
        )

    idx = np.where(scale == 0)  # plot vessel which scale is 0.2 by finding its indexs
    plt.figure()
    plt.scatter(x[idx], y[idx])
    plt.axis("equal")
    plt.savefig(osp.join(PLOT_DIR, "one_scale_sample"), bbox_inches="tight")

    np.savez("x_0526_pd", x=x)
    np.savez("y_0526_pd", y=y)
    np.savez("scale_0526_pd", scale=scale)
    interior_geom = ppsci.geometry.PointCloud(
        interior={"x": x, "y": y, "scale": scale},
        coord_keys=["x", "y", "scale"],
    )

    def init_func(m):
        if misc.typename(m) == "Linear":
            ppsci.utils.initializer.kaiming_normal_(m.weight, reverse=True)

    model_1 = ppsci.arch.MLP(("x", "y", "scale"), ("u"), 3, 20, "swish")
    model_2 = ppsci.arch.MLP(("x", "y", "scale"), ("v"), 3, 20, "swish")
    model_3 = ppsci.arch.MLP(("x", "y", "scale"), ("p"), 3, 20, "swish")
    model_1.apply(init_func)
    model_2.apply(init_func)
    model_3.apply(init_func)

    # print(f"layer 1 mean : {np.mean(model_1.linears[0].weight.numpy())}")
    # print(f"layer 1 var : {np.var(model_1.linears[0].weight.numpy())}")

    # print(f"layer 2 mean : {np.mean(model_1.linears[1].weight.numpy())}")
    # print(f"layer 2 var : {np.var(model_1.linears[1].weight.numpy())}")

    # print(f"layer 3 mean : {np.mean(model_1.linears[2].weight.numpy())}")
    # print(f"layer 3 var : {np.var(model_1.linears[2].weight.numpy())}")

    # print(f"layer 4 mean : {np.mean(model_1.last_fc.weight.numpy())}")
    # print(f"layer 4 var : {np.var(model_1.last_fc.weight.numpy())}")

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
                    / np.sqrt(2 * np.pi * SIGMA**2)
                    * paddle.exp(-((x - mu) ** 2) / (2 * SIGMA**2))
                )
                self.h = R_INLET - R
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
                ValueError(f"{next(iter(out.keys()))} is not a valid key.")

            return new_out

    shared_transform = Output_transform()
    model_1.register_output_transform(shared_transform)
    model_2.register_output_transform(shared_transform)
    model_3.register_output_transform(shared_transform)
    model = ppsci.arch.ModelList((model_1, model_2, model_3))

    optimizer_1 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=1e-15
    )((model_1,))
    optimizer_2 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=1e-15
    )((model_2,))
    optimizer_3 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=1e-15
    )((model_3,))
    optimizer = ppsci.optimizer.OptimizerList((optimizer_1, optimizer_2, optimizer_3))

    equation = {"NavierStokes": ppsci.equation.NavierStokes(NU, RHO, 2, False)}

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
        name="EQ",
    )
    constrain_dict = {pde_constraint.name: pde_constraint}
    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constrain_dict,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=int(x.shape[0] / BATCH_SIZE),
        eval_during_train=False,
        save_freq=10,
        log_freq=1,
        equation=equation,
    )

    solver.train()

    def single_test(x, y, scale, solver):
        xt = paddle.to_tensor(x)
        yt = paddle.to_tensor(y)
        scalet = scale * paddle.ones_like(xt)
        net_in = {"x": xt, "y": yt, "scale": scalet}
        output_dict = solver.predict(net_in, 100)
        return output_dict

    scale_test = np.load("./data/aneurysm_scale0005to002_eval0to002mean001_3sigma.npz")[
        "scale"
    ]
    CASE_COUNT = [1.0, 151.0, 486.0]
    w_ctl = np.zeros([len(scale_test), 1])
    w_ctl_Ml = np.zeros([len(scale_test), 1])
    PLOT_X = 0.8
    PLOT_Y = 0.06
    FONTSIZE = 14
    axis_limit = [0, 1, -0.15, 0.15]
    path = "./data/cases/"
    D_P = 0.1
    error_u = []
    error_v = []
    N_CL = 200  # number of sampling points in centerline (confused about centerline, but the paper did not explain)
    x_centerline = np.linspace(
        X_IN, X_OUT, N_CL, dtype=paddle.get_default_dtype()
    ).reshape(N_CL, 1)
    y_centerline = np.zeros_like(x_centerline, dtype=paddle.get_default_dtype())
    for caseIdx in CASE_COUNT:
        scale = scale_test[int(caseIdx - 1)]
        data_CFD = np.load(osp.join(path, str(caseIdx) + "CFD_contour.npz"))
        x = data_CFD["x"].astype(paddle.get_default_dtype())
        y = data_CFD["y"].astype(paddle.get_default_dtype())
        u_cfd = data_CFD["U"].astype(paddle.get_default_dtype())
        # p_cfd = data_CFD["P"].astype(paddle.get_default_dtype()) # missing data

        n = len(x)
        output_dict = single_test(
            x.reshape(n, 1),
            y.reshape(n, 1),
            np.ones((n, 1), dtype=paddle.get_default_dtype()) * scale,
            solver,
        )
        u, v, p = output_dict["u"], output_dict["v"], output_dict["p"]
        w = np.zeros_like(u)
        u_vec = np.concatenate([u, v, w], axis=1)
        error_u.append(
            np.linalg.norm(u_vec[:, 0] - u_cfd[:, 0]) / (D_P * len(u_vec[:, 0]))
        )
        error_v.append(
            np.linalg.norm(u_vec[:, 1] - u_cfd[:, 1]) / (D_P * len(u_vec[:, 0]))
        )
        # error_p = np.linalg.norm(p - p_cfd) / (D_P * D_P)

        # Streamwise velocity component u
        plt.figure()
        plt.subplot(212)
        plt.scatter(x, y, c=u_vec[:, 0], vmin=min(u_cfd[:, 0]), vmax=max(u_cfd[:, 0]))
        plt.text(PLOT_X, PLOT_Y, r"DNN", {"color": "b", "fontsize": FONTSIZE})
        plt.axis(axis_limit)
        plt.colorbar()
        plt.subplot(211)
        plt.scatter(x, y, c=u_cfd[:, 0], vmin=min(u_cfd[:, 0]), vmax=max(u_cfd[:, 0]))
        plt.colorbar()
        plt.text(PLOT_X, PLOT_Y, r"CFD", {"color": "b", "fontsize": FONTSIZE})
        plt.axis(axis_limit)
        plt.savefig(
            osp.join(PLOT_DIR, f"{int(caseIdx)}_scale_{scale}_uContour_test.png"),
            bbox_inches="tight",
        )

        # Spanwise velocity component v
        plt.figure()
        plt.subplot(212)
        plt.scatter(x, y, c=u_vec[:, 1], vmin=min(u_cfd[:, 1]), vmax=max(u_cfd[:, 1]))
        plt.text(PLOT_X, PLOT_Y, r"DNN", {"color": "b", "fontsize": FONTSIZE})
        plt.axis(axis_limit)
        plt.colorbar()
        plt.subplot(211)
        plt.scatter(x, y, c=u_cfd[:, 1], vmin=min(u_cfd[:, 1]), vmax=max(u_cfd[:, 1]))
        plt.colorbar()
        plt.text(PLOT_X, PLOT_Y, r"CFD", {"color": "b", "fontsize": FONTSIZE})
        plt.axis(axis_limit)
        plt.savefig(
            osp.join(PLOT_DIR, f"{int(caseIdx)}_scale_{scale}_vContour_test.png"),
            bbox_inches="tight",
        )
        plt.close("all")

        # Centerline wall shear profile tau_c (downside)
        data_CFD_wss = np.load(path + str(caseIdx) + "CFD_wss.npz")
        x_inital = data_CFD_wss["x"]
        wall_shear_mag_up = data_CFD_wss["wss"]

        D_H = 0.001  # The spanwise distance is approximately the height of the wall
        r_cl = (
            scale
            * 1
            / np.sqrt(2 * np.pi * SIGMA**2)
            * np.exp(-((x_centerline - mu) ** 2) / (2 * SIGMA**2))
        )
        y_wall = (-R_INLET + D_H) * np.ones_like(
            x_centerline, dtype=paddle.get_default_dtype()
        ) + r_cl
        output_dict_wss = single_test(
            x_centerline, y_wall, np.ones((N_CL, 1)) * scale, solver
        )
        v_cl_total = np.zeros_like(
            x_centerline, dtype=paddle.get_default_dtype()
        )  # assuming normal velocity along the wall is zero
        u_cl = output_dict_wss["u"].numpy()
        v_cl = output_dict_wss["v"].numpy()
        for i in range(N_CL):
            v_cl_total[i] = np.sqrt(u_cl[i] ** 2 + v_cl[i] ** 2)
        tau_c = NU * v_cl_total / D_H

        plt.figure()
        plt.plot(
            x_inital,
            wall_shear_mag_up,
            label="CFD",
            color="darkblue",
            linestyle="-",
            lw=3.0,
            alpha=1.0,
        )
        plt.plot(
            x_inital,
            tau_c,
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
            osp.join(PLOT_DIR, f"{int(caseIdx)}_nu__{scale}_wallshear_test.png"),
            bbox_inches="tight",
        )
        plt.close("all")
    print(f"epochs = {EPOCHS}")
    print(f"Table 1 : Aneurysm - Geometry error u : {sum(error_u) / len(error_u): .3e}")
    print(f"Table 1 : Aneurysm - Geometry error v : {sum(error_v) / len(error_v): .3e}")

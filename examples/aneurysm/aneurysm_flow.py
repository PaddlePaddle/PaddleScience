import math
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.fluid import core

import ppsci
from ppsci.utils import misc


def generate_scale_for_all_x(x, scale):
    new_scale = np.tile(scale, (len(x), 1))
    new_x = np.array([np.tile(val, len(scale)) for val in x]).reshape(len(new_scale), 1)
    return new_x, new_scale


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
    import os

    os.chdir("/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm")
    # set output directory
    OUTPUT_DIR = "./output_0531"
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
    N_y = 20
    x_inital = np.linspace(X_IN, X_OUT, 100, dtype=paddle.get_default_dtype()).reshape(
        100, 1
    )
    mu = 0.5 * (X_OUT - X_IN)
    x_20_copy = np.tile(x_inital, (20, 1))  # duplicate 20 times of x for dataloader
    SIGMA = 0.1
    SCALE_START = -0.02
    SCALE_END = 0
    scale_initial = np.linspace(
        SCALE_START, SCALE_END, 50, endpoint=True, dtype=paddle.get_default_dtype()
    ).reshape(50, 1)
    x, scale = generate_scale_for_all_x(
        x_20_copy, scale_initial
    )  # generate 50 scale factors for each x (mesh grid)

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

    # index = [i for i in range(x.shape[0])]
    # res = list(zip(x, y, scale))
    # np.random.shuffle(res)
    # x, y, scale = zip(*res)fastype
    # x = np.array(x).astype(float)
    # y = np.array(y).astype(float)
    # scale = np.array(scale).astype(float)

    np.savez("x_0526_pd", x=x)
    np.savez("y_0526_pd", y=y)
    np.savez("scale_0526_pd", scale=scale)
    interior_geom = ppsci.geometry.PointCloud(
        interior={"x": x, "y": y, "scale": scale},
        coord_keys=["x", "y", "scale"],
    )

    def _calculate_fan_in_and_fan_out(tensor):
        if tensor.ndim < 2:
            raise ValueError(
                f"tensor.ndim should be no less than 2, but got {tensor.ndim}."
            )
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.ndim > 2:
            receptive_field_size = np.prod(tensor.shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    def init_func(m):
        if misc.typename(m) == "Linear":
            ppsci.utils.initializer.kaiming_normal_(m.weight, reverse=True)

    model_1 = ppsci.arch.MLP(
        ["x", "y", "scale"],
        ["u"],
        3,
        20,
        "swish",
        False,
        False,
        np.load(
            f"/workspace/wangguan/LabelFree-DNN-Surrogate/ParametricAneurysm/net_params/net2_params/weight_0_0.npz"
        ),
        np.load(
            f"/workspace/wangguan/LabelFree-DNN-Surrogate/ParametricAneurysm/net_params/net2_params/bias_0_0.npz"
        ),
    )

    model_2 = ppsci.arch.MLP(
        ["x", "y", "scale"],
        ["v"],
        3,
        20,
        "swish",
        False,
        False,
        np.load(
            f"/workspace/wangguan/LabelFree-DNN-Surrogate/ParametricAneurysm/net_params/net3_params/weight_0_0.npz"
        ),
        np.load(
            f"/workspace/wangguan/LabelFree-DNN-Surrogate/ParametricAneurysm/net_params/net3_params/bias_0_0.npz"
        ),
    )

    model_3 = ppsci.arch.MLP(
        ["x", "y", "scale"],
        ["p"],
        3,
        20,
        "swish",
        False,
        False,
        np.load(
            f"/workspace/wangguan/LabelFree-DNN-Surrogate/ParametricAneurysm/net_params/net4_params/weight_0_0.npz"
        ),
        np.load(
            f"/workspace/wangguan/LabelFree-DNN-Surrogate/ParametricAneurysm/net_params/net4_params/bias_0_0.npz"
        ),
    )
    # model_1.apply(init_func)
    # model_2.apply(init_func)
    # model_3.apply(init_func)

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
                raise NotImplementedError(f"{out.keys()} are outputs to be implemented")

            return new_out

    shared_transform = Output_transform()
    model_1.register_output_transform(shared_transform)
    model_2.register_output_transform(shared_transform)
    model_3.register_output_transform(shared_transform)
    model = ppsci.arch.ModelList([model_1, model_2, model_3])

    optimizer_1 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=10**-15
    )([model_1])
    optimizer_2 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=10**-15
    )([model_2])
    optimizer_3 = ppsci.optimizer.Adam(
        LEARNING_RATE, beta1=0.9, beta2=0.99, epsilon=10**-15
    )([model_3])
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
                "shuffle": True,
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
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        iters_per_epoch=int(x.shape[0] / BATCH_SIZE),
        eval_during_train=False,
        save_freq=10,
        log_freq=100,
        equation=equation,
        checkpoint_path="/workspace/wangguan/PaddleScience_Surrogate/examples/aneurysm/output_0531/checkpoints/epoch_30",
    )

    # solver.train()

    def single_test(x, y, scale, solver):
        xt = paddle.to_tensor(x, dtype="float32")
        yt = paddle.to_tensor(y, dtype="float32")
        scalet = scale * paddle.ones_like(xt)
        net_in = {"x": xt, "y": yt, "scale": scalet}
        output_dict = solver.predict(net_in, 10000)
        return output_dict

    scale_test = np.load("./data/aneurysm_scale0005to002_eval0to002mean001_3sigma.npz")[
        "scale"
    ]
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
        # Data_NN = np.load(path + str(caseIdx) + "NN_contour.npz")
        x = Data_CFD["x"]
        y = Data_CFD["y"]
        U_CFD = Data_CFD["U"]
        # U = Data_NN["U"]
        n = len(x)
        output_dict = single_test(
            x.reshape(n, 1), y.reshape(n, 1), np.ones((n, 1)) * scale, solver
        )
        u, v, p = output_dict["u"], output_dict["v"], output_dict["p"]
        w = np.zeros_like(u)
        U = np.concatenate([u, v, w], axis=1)

        # Streamwise velocity component u
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
            osp.join(PLOT_DIR, f"{int(caseIdx)}_scale_{scale}_uContour_test.png"),
            bbox_inches="tight",
        )

        # Spanwise velocity component v
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
            osp.join(PLOT_DIR, f"{int(caseIdx)}_scale_{scale}_vContour_test.png"),
            bbox_inches="tight",
        )
        plt.close("all")

        # Centerline wall shear profile tau_c
        Data_CFD_wss = np.load(path + str(caseIdx) + "CFD_wss.npz")
        x_inital = Data_CFD_wss["x"]
        wall_shear_mag_up = Data_CFD_wss["wss"]
        Data_NN_wss = np.load(path + str(caseIdx) + "NN_wss.npz")
        NNwall_shear_mag_up = Data_NN_wss["wss"]

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
            osp.join(PLOT_DIR, f"{int(caseIdx)}_nu__{scale}_wallshear_test.png"),
            bbox_inches="tight",
        )
        plt.close("all")

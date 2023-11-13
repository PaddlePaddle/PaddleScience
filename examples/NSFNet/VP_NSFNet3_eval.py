import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger

paddle.set_default_dtype("float32")


def data_generate(x, y, z, t):

    a, d = 1, 1
    u = (
        -a
        * (
            np.exp(a * x) * np.sin(a * y + d * z)
            + np.exp(a * z) * np.cos(a * x + d * y)
        )
        * np.exp(-d * d * t)
    )
    v = (
        -a
        * (
            np.exp(a * y) * np.sin(a * z + d * x)
            + np.exp(a * x) * np.cos(a * y + d * z)
        )
        * np.exp(-d * d * t)
    )
    w = (
        -a
        * (
            np.exp(a * z) * np.sin(a * x + d * y)
            + np.exp(a * y) * np.cos(a * z + d * x)
        )
        * np.exp(-d * d * t)
    )
    p = (
        -0.5
        * a
        * a
        * (
            np.exp(2 * a * x)
            + np.exp(2 * a * y)
            + np.exp(2 * a * z)
            + 2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z))
            + 2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x))
            + 2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))
        )
        * np.exp(-2 * d * d * t)
    )

    return u, v, w, p


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet3.yaml")
def main(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)

    # set model
    input_key = ("x", "y", "z", "t")
    output_key = ("u", "v", "w", "p")
    model = ppsci.arch.MLP(
        input_key,
        output_key,
        cfg.model.ihlayers,
        cfg.model.ineurons,
        "tanh",
        input_dim=len(input_key),
        output_dim=len(output_key),
        Xavier=True,
    )

    # set the number of residual samples
    N_TRAIN = cfg.ntrain

    # unsupervised part
    xx = np.random.randint(31, size=N_TRAIN) / 15 - 1
    yy = np.random.randint(31, size=N_TRAIN) / 15 - 1
    zz = np.random.randint(31, size=N_TRAIN) / 15 - 1
    tt = np.random.randint(11, size=N_TRAIN) / 10

    x_train = xx.reshape(xx.shape[0], 1).astype("float32")
    y_train = yy.reshape(yy.shape[0], 1).astype("float32")
    z_train = zz.reshape(zz.shape[0], 1).astype("float32")
    t_train = tt.reshape(tt.shape[0], 1).astype("float32")

    # test data
    x_star = ((np.random.rand(1000, 1) - 1 / 2) * 2).astype("float32")
    y_star = ((np.random.rand(1000, 1) - 1 / 2) * 2).astype("float32")
    z_star = ((np.random.rand(1000, 1) - 1 / 2) * 2).astype("float32")
    t_star = (np.random.randint(11, size=(1000, 1)) / 10).astype("float32")

    u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)

    valida_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x_star, "y": y_star, "z": z_star, "t": t_star},
            "label": {"u": u_star, "v": v_star, "w": w_star, "p": p_star},
        },
        "total_size": u_star.shape[0],
        "batch_size": u_star.shape[0],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    geom = ppsci.geometry.PointCloud(
        {"x": x_train, "y": y_train, "z": z_train, "t": t_train}, ("x", "y", "z", "t")
    )

    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu=1.0 / cfg.Re, rho=1.0, dim=3, time=True
        ),
    }
    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # wrap validator
    validator = {residual_validator.name: residual_validator}

    # load solver
    solver = ppsci.solver.Solver(
        model,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.pretrained_model_path,  ### the path of the model
    )

    # print the relative error
    us = []
    vs = []
    ws = []
    for i in [0, 0.25, 0.5, 0.75, 1.0]:
        x_star, y_star, z_star = np.mgrid[-1.0:1.0:100j, -1.0:1.0:100j, -1.0:1.0:100j]
        x_star, y_star, z_star = (
            x_star.reshape(-1, 1),
            y_star.reshape(-1, 1),
            z_star.reshape(-1, 1),
        )
        t_star = i * np.ones(x_star.shape)
        u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)

        solution = solver.predict({"x": x_star, "y": y_star, "z": z_star, "t": t_star})
        u_pred = solution["u"]
        v_pred = solution["v"]
        w_pred = solution["w"]
        p_pred = solution["p"]
        p_pred = p_pred - p_pred.mean() + p_star.mean()
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
        error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
        error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
        us.append(error_u)
        vs.append(error_v)
        ws.append(error_w)
        print("t={:.2f},relative error of u: {:.3e}".format(t_star[0].item(), error_u))
        print("t={:.2f},relative error of v: {:.3e}".format(t_star[0].item(), error_v))
        print("t={:.2f},relative error of w: {:.3e}".format(t_star[0].item(), error_w))
        print("t={:.2f},relative error of p: {:.3e}".format(t_star[0].item(), error_p))

    ## plot vorticity
    grid_x, grid_y = np.mgrid[-1.0:1.0:1000j, -1.0:1.0:1000j]
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)
    grid_z = np.zeros(grid_x.shape)
    T = np.linspace(0, 1, 20)
    for i in T:
        t_star = i * np.ones(x_star.shape)
        u_star, v_star, w_star, p_star = data_generate(grid_x, grid_y, grid_z, t_star)

        solution = solver.predict({"x": grid_x, "y": grid_y, "z": grid_z, "t": t_star})
        u_pred = np.array(solution["u"])
        v_pred = np.array(solution["v"])
        w_pred = np.array(solution["w"])
        p_pred = p_pred - p_pred.mean() + p_star.mean()
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        ax[0, 0].contourf(
            grid_x.reshape(1000, 1000),
            grid_y.reshape(1000, 1000),
            u_star.reshape(1000, 1000),
            cmap=plt.get_cmap("RdYlBu"),
        )
        ax[0, 1].contourf(
            grid_x.reshape(1000, 1000),
            grid_y.reshape(1000, 1000),
            u_pred.reshape(1000, 1000),
            cmap=plt.get_cmap("RdYlBu"),
        )
        ax[1, 0].contourf(
            grid_x.reshape(1000, 1000),
            grid_y.reshape(1000, 1000),
            v_star.reshape(1000, 1000),
            cmap=plt.get_cmap("RdYlBu"),
        )
        ax[1, 1].contourf(
            grid_x.reshape(1000, 1000),
            grid_y.reshape(1000, 1000),
            v_pred.reshape(1000, 1000),
            cmap=plt.get_cmap("RdYlBu"),
        )
        ax[2, 0].contourf(
            grid_x.reshape(1000, 1000),
            grid_y.reshape(1000, 1000),
            w_star.reshape(1000, 1000),
            cmap=plt.get_cmap("RdYlBu"),
        )
        ax[2, 1].contourf(
            grid_x.reshape(1000, 1000),
            grid_y.reshape(1000, 1000),
            w_pred.reshape(1000, 1000),
            cmap=plt.get_cmap("RdYlBu"),
        )
        ax[0, 0].set_title("u_exact")
        ax[0, 1].set_title("u_pred")
        ax[1, 0].set_title("v_exact")
        ax[1, 1].set_title("v_pred")
        ax[2, 0].set_title("w_exact")
        ax[2, 1].set_title("w_pred")


if __name__ == "__main__":
    main()

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def analytic_solution_generate(x, y, z, t):
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


def generate_data(N_TRAIN):
    # generate boundary data
    x1 = np.linspace(-1, 1, 31)
    y1 = np.linspace(-1, 1, 31)
    z1 = np.linspace(-1, 1, 31)
    t1 = np.linspace(0, 1, 11)
    b0 = np.array([-1] * 900)
    b1 = np.array([1] * 900)

    xt = np.tile(x1[0:30], 30)
    yt = np.tile(y1[0:30], 30)
    xt1 = np.tile(x1[1:31], 30)
    yt1 = np.tile(y1[1:31], 30)

    yr = y1[0:30].repeat(30)
    zr = z1[0:30].repeat(30)
    yr1 = y1[1:31].repeat(30)
    zr1 = z1[1:31].repeat(30)

    train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
    train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
    train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
    train1t = np.tile(t1, 5400)

    train1ub, train1vb, train1wb, train1pb = analytic_solution_generate(
        train1x, train1y, train1z, train1t
    )

    xb_train = train1x.reshape(train1x.shape[0], 1).astype("float32")
    yb_train = train1y.reshape(train1y.shape[0], 1).astype("float32")
    zb_train = train1z.reshape(train1z.shape[0], 1).astype("float32")
    tb_train = train1t.reshape(train1t.shape[0], 1).astype("float32")
    ub_train = train1ub.reshape(train1ub.shape[0], 1).astype("float32")
    vb_train = train1vb.reshape(train1vb.shape[0], 1).astype("float32")
    wb_train = train1wb.reshape(train1wb.shape[0], 1).astype("float32")

    # generate initial data
    x_0 = np.tile(x1, 31 * 31)
    y_0 = np.tile(y1.repeat(31), 31)
    z_0 = z1.repeat(31 * 31)
    t_0 = np.array([0] * x_0.shape[0])
    u_0, v_0, w_0, p_0 = analytic_solution_generate(x_0, y_0, z_0, t_0)
    u0_train = u_0.reshape(u_0.shape[0], 1).astype("float32")
    v0_train = v_0.reshape(v_0.shape[0], 1).astype("float32")
    w0_train = w_0.reshape(w_0.shape[0], 1).astype("float32")
    x0_train = x_0.reshape(x_0.shape[0], 1).astype("float32")
    y0_train = y_0.reshape(y_0.shape[0], 1).astype("float32")
    z0_train = z_0.reshape(z_0.shape[0], 1).astype("float32")
    t0_train = t_0.reshape(t_0.shape[0], 1).astype("float32")

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

    u_star, v_star, w_star, p_star = analytic_solution_generate(
        x_star, y_star, z_star, t_star
    )

    return (
        x_train,
        y_train,
        z_train,
        t_train,
        x0_train,
        y0_train,
        z0_train,
        t0_train,
        u0_train,
        v0_train,
        w0_train,
        xb_train,
        yb_train,
        zb_train,
        tb_train,
        ub_train,
        vb_train,
        wb_train,
        x_star,
        y_star,
        z_star,
        t_star,
        u_star,
        v_star,
        w_star,
        p_star,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet3.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


def train(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    ITERS_PER_EPOCH = cfg.iters_per_epoch

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set the number of residual samples
    N_TRAIN = cfg.ntrain

    # set the number of boundary samples
    NB_TRAIN = cfg.nb_train

    # set the number of initial samples
    N0_TRAIN = cfg.n0_train
    ALPHA = cfg.alpha
    BETA = cfg.beta
    (
        x_train,
        y_train,
        z_train,
        t_train,
        x0_train,
        y0_train,
        z0_train,
        t0_train,
        u0_train,
        v0_train,
        w0_train,
        xb_train,
        yb_train,
        zb_train,
        tb_train,
        ub_train,
        vb_train,
        wb_train,
        x_star,
        y_star,
        z_star,
        t_star,
        u_star,
        v_star,
        w_star,
        p_star,
    ) = generate_data(N_TRAIN)

    # set dataloader config
    train_dataloader_cfg_b = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": xb_train, "y": yb_train, "z": zb_train, "t": tb_train},
            "label": {"u": ub_train, "v": vb_train, "w": wb_train},
        },
        "batch_size": NB_TRAIN,
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    train_dataloader_cfg_0 = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x0_train, "y": y0_train, "z": z0_train, "t": t0_train},
            "label": {"u": u0_train, "v": v0_train, "w": w0_train},
        },
        "batch_size": N0_TRAIN,
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

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

    # supervised constraint s.t ||u-u_b||
    sup_constraint_b = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_b,
        ppsci.loss.MSELoss("mean", ALPHA),
        name="Sup_b",
    )

    # supervised constraint s.t ||u-u_0||
    sup_constraint_0 = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_0,
        ppsci.loss.MSELoss("mean", BETA),
        name="Sup_0",
    )

    # set equation constarint s.t. ||F(u)||
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu=1.0 / cfg.re, rho=1.0, dim=3, time=True
        ),
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom,
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": N_TRAIN,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    # wrap constraint
    constraint = {
        pde_constraint.name: pde_constraint,
        sup_constraint_b.name: sup_constraint_b,
        sup_constraint_0.name: sup_constraint_0,
    }

    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        output_expr={
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "p": lambda d: d["p"] - d["p"].min() + p_star.min(),
        },
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # wrap validator
    validator = {residual_validator.name: residual_validator}

    # set optimizer
    epoch_list = [5000, 5000, 50000, 50000]
    new_epoch_list = []
    for i, _ in enumerate(epoch_list):
        new_epoch_list.append(sum(epoch_list[: i + 1]))
    EPOCHS = new_epoch_list[-1]
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    lr_scheduler = ppsci.optimizer.lr_scheduler.Piecewise(
        EPOCHS, ITERS_PER_EPOCH, new_epoch_list, lr_list
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        optimizer=optimizer,
        epochs=EPOCHS,
        lr_scheduler=lr_scheduler,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=True,
        log_freq=cfg.log_freq,
        eval_freq=cfg.eval_freq,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=None,
        eval_with_no_grad=False,
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()
    solver.plot_loss_history()


def evaluate(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)
    ppsci.utils.load_pretrain(model, cfg.pretrained_model_path)

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

    u_star, v_star, w_star, p_star = analytic_solution_generate(
        x_star, y_star, z_star, t_star
    )

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
            nu=1.0 / cfg.re, rho=1.0, dim=3, time=True
        ),
    }
    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        output_expr={
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "p": lambda d: d["p"] - d["p"].min() + p_star.min(),
        },
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
        u_star, v_star, w_star, p_star = analytic_solution_generate(
            x_star, y_star, z_star, t_star
        )

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
    T = np.linspace(0, 1, 101)
    for i in T:
        t_star = i * np.ones(x_star.shape)
        u_star, v_star, w_star, p_star = analytic_solution_generate(
            grid_x, grid_y, grid_z, t_star
        )

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
        time = "%.3f" % i
        fig.savefig(OUTPUT_DIR + f"/velocity_t={str(time)}.png")


if __name__ == "__main__":
    main()

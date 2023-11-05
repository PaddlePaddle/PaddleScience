import ppsci
import paddle

paddle.set_default_dtype("float32")
import numpy as np
from ppsci.utils import logger
import scipy


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet2.yaml")
def main(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    ITERS_PER_EPOCH = cfg.iters_per_epoch
    # set model
    input_key = ("x", "y", "t")
    output_key = ("u", "v", "p")
    model = ppsci.arch.MLP(
        input_key, output_key, cfg.model.ihlayers, cfg.model.ineurons, "tanh", input_dim=len(input_key),
        output_dim=len(output_key), Xavier=True
    )

    ## set the number of residual samples
    N_TRAIN = cfg.ntrain

    ## set the number of boundary samples
    NB_TRAIN = cfg.nb_train

    ## set the number of initial samples
    N0_TRAIN = cfg.n0_train

    data = scipy.io.loadmat(cfg.data_path)

    U_star = data['U_star'].astype('float32')  # N x 2 x T
    P_star = data['p_star'].astype('float32')  # N x T
    t_star = data['t'].astype('float32')  # T x 1
    X_star = data['X_star'].astype('float32')  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data

    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    # need add unsupervised part

    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]

    data_t0 = data_domain[:, :][data_domain[:, 2] == 0]

    data_y1 = data_domain[:, :][data_domain[:, 0] == 1]
    data_y8 = data_domain[:, :][data_domain[:, 0] == 8]
    data_x = data_domain[:, :][data_domain[:, 1] == -2]
    data_x2 = data_domain[:, :][data_domain[:, 1] == 2]

    data_sup_b_train = np.concatenate([data_y1, data_y8, data_x, data_x2], 0)

    idx = np.random.choice(data_domain.shape[0], N_TRAIN, replace=False)

    x_train = data_domain[idx, 0].reshape(data_domain[idx, 0].shape[0], 1)
    y_train = data_domain[idx, 1].reshape(data_domain[idx, 1].shape[0], 1)
    t_train = data_domain[idx, 2].reshape(data_domain[idx, 2].shape[0], 1)

    x0_train = data_t0[:, 0].reshape(data_t0[:, 0].shape[0], 1)
    y0_train = data_t0[:, 1].reshape(data_t0[:, 1].shape[0], 1)
    t0_train = data_t0[:, 2].reshape(data_t0[:, 2].shape[0], 1)
    u0_train = data_t0[:, 3].reshape(data_t0[:, 3].shape[0], 1)
    v0_train = data_t0[:, 4].reshape(data_t0[:, 4].shape[0], 1)

    xb_train = data_sup_b_train[:, 0].reshape(data_sup_b_train[:, 0].shape[0], 1)
    yb_train = data_sup_b_train[:, 1].reshape(data_sup_b_train[:, 1].shape[0], 1)
    tb_train = data_sup_b_train[:, 2].reshape(data_sup_b_train[:, 2].shape[0], 1)
    ub_train = data_sup_b_train[:, 3].reshape(data_sup_b_train[:, 3].shape[0], 1)
    vb_train = data_sup_b_train[:, 4].reshape(data_sup_b_train[:, 4].shape[0], 1)

    # set test set
    snap = np.array([0])
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    # set dataloader config
    train_dataloader_cfg_b = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": xb_train, "y": yb_train, "t": tb_train},
            "label": {"u": ub_train, "v": vb_train}
        },
        "batch_size": NB_TRAIN,
        'iters_per_epoch': ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    train_dataloader_cfg_0 = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x0_train, "y": y0_train, "t": t0_train},
            "label": {"u": u0_train, "v": v0_train}
        },
        "batch_size": N0_TRAIN,
        'iters_per_epoch': ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    valida_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x_star, "y": y_star, "t": t_star},
            "label": {"u": u_star, "v": v_star, "p": p_star}},
        'total_size': u_star.shape[0],
        "batch_size": u_star.shape[0],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    geom = ppsci.geometry.PointCloud({"x": x_train, "y": y_train, "t": t_train}, ("x", "y", "t"))

    ## supervised constraint s.t ||u-u_b||
    sup_constraint_b = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_b,
        ppsci.loss.MSELoss("mean"),
        name="Sup_b",
    )

    ## supervised constraint s.t ||u-u_0||
    sup_constraint_0 = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_0,
        ppsci.loss.MSELoss("mean"),
        name="Sup_0",
    )

    # set equation constarint s.t. ||F(u)||
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(nu=1.0 / cfg.re, rho=1.0, dim=2, time=True),
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom,
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": N_TRAIN,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    constraint = {pde_constraint.name: pde_constraint, sup_constraint_b.name: sup_constraint_b,
                  sup_constraint_0.name: sup_constraint_0}

    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # wrap validator
    validator = {residual_validator.name: residual_validator}

    # set optimizer
    epoch_list = [5000, 5000, 50000, 50000]
    new_epoch_list = []
    for i, _ in enumerate(epoch_list):
        new_epoch_list.append(sum(epoch_list[:i + 1]))
    EPOCHS = new_epoch_list[-1]
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    lr_scheduler = ppsci.optimizer.lr_scheduler.Piecewise(EPOCHS, ITERS_PER_EPOCH, new_epoch_list, lr_list)()
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
        output_dir='/home/aistudio/'
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()

    solver.plot_loss_history()


if __name__ == "__main__":
    main()

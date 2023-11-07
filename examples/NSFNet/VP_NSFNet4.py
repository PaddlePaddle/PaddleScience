import ppsci
import paddle

paddle.set_default_dtype("float32")
import numpy as np
from ppsci.utils import logger
import scipy


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet3.yaml")
def main(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    ITERS_PER_EPOCH = cfg.iters_per_epoch
    # set model
    input_key = ("x", "y", "z", "t")
    output_key = ("u", "v", "w", "p")
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

    ALPHA = cfg.alpha
    BETA = cfg.beta

    # Load Data
    train_ini1 = np.load(cfg.train_ini2)
    train_iniv1 = np.load(cfg.train_iniv2)
    train_xb1 = np.load(cfg.train_xb2.npy)
    train_vb1 = np.load(cfg.train_vb2.npy)

    xnode = np.linspace(12.47, 12.66, 191)
    ynode = np.linspace(-1, -0.0031, 998)
    znode = np.linspace(4.61, 4.82, 211)

    x0_train = train_ini1[:, 0:1]
    y0_train = train_ini1[:, 1:2]
    z0_train = train_ini1[:, 2:3]
    t0_train = np.zeros(train_ini1[:, 0:1].shape, np.float32)
    u0_train = train_iniv1[:, 0:1]
    v0_train = train_iniv1[:, 1:2]
    w0_train = train_iniv1[:, 2:3]

    xb_train = train_xb1[:, 0:1]
    yb_train = train_xb1[:, 1:2]
    zb_train = train_xb1[:, 2:3]
    tb_train = train_xb1[:, 3:4]
    ub_train = train_vb1[:, 0:1]
    vb_train = train_vb1[:, 1:2]
    wb_train = train_vb1[:, 2:3]

    x_train1 = xnode.reshape(-1, 1)[np.random.choice(191, 100000, replace=True), :]
    y_train1 = ynode.reshape(-1, 1)[np.random.choice(998, 100000, replace=True), :]
    z_train1 = znode.reshape(-1, 1)[np.random.choice(211, 100000, replace=True), :]
    x_train = np.tile(x_train1, (17, 1))
    y_train = np.tile(y_train1, (17, 1))
    z_train = np.tile(z_train1, (17, 1))

    total_times1 = np.array(list(range(17))) * 0.0065
    t_train1 = total_times1.repeat(100000)
    t_train = t_train1.reshape(-1, 1)

    # Test Data
    test_x = np.load(cfg.test43_l.npy)
    test_v = np.load(cfg.test43_vp)
    t = np.array([0.0065, 4 * 0.0065, 7 * 0.0065, 10 * 0.0065, 13 * 0.0065])
    t_star = np.tile(t.reshape(5, 1), (1, 3000)).reshape(-1, 1)
    x_star = np.tile(test_x[:, 0:1], (5, 1))
    y_star = np.tile(test_x[:, 1:2], (5, 1))
    z_star = np.tile(test_x[:, 2:3], (5, 1))
    u_star = test_v[:, 0:1]
    v_star = test_v[:, 1:2]
    w_star = test_v[:, 2:3]
    p_star = test_v[:, 3:4]

    # set dataloader config
    train_dataloader_cfg_b = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": xb_train, "y": yb_train, "z": zb_train, "t": tb_train},
            "label": {"u": ub_train, "v": vb_train, "w": wb_train}
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
            "input": {"x": x0_train, "y": y0_train, "z": z0_train, "t": t0_train},
            "label": {"u": u0_train, "v": v0_train, "w": w0_train}
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
            "input": {"x": x_star, "y": y_star, "z": z_star, "t": t_star},
            "label": {"u": u_star, "v": v_star, "w": w_star, "p": p_star}},
        'total_size': u_star.shape[0],
        "batch_size": u_star.shape[0],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    geom = ppsci.geometry.PointCloud({"x": x_train, "y": y_train, "z": z_train, "t": t_train}, ("x", "y", "z", "t"))

    ## supervised constraint s.t ||u-u_b||
    sup_constraint_b = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_b,
        ppsci.loss.MSELoss("mean", ALPHA),
        name="Sup_b",
    )

    ## supervised constraint s.t ||u-u_0||
    sup_constraint_0 = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_0,
        ppsci.loss.MSELoss("mean", BETA),
        name="Sup_0",
    )

    # set equation constarint s.t. ||F(u)||
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(nu=1.0 / cfg.Re, rho=1.0, dim=3, time=True),
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

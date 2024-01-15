from os import path as osp

import hydra
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    inverse_lambda_net = ppsci.arch.MLP(**cfg.MODEL.inverse_lambda_net)
    inverse_mu_net = ppsci.arch.MLP(**cfg.MODEL.inverse_mu_net)
    # freeze models
    disp_net.freeze()
    stress_net.freeze()
    # wrap to a model_list
    model = ppsci.arch.ModelList(
        (disp_net, stress_net, inverse_lambda_net, inverse_mu_net)
    )

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((inverse_lambda_net, inverse_mu_net))

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_="lambda_", mu="mu", dim=3
        )
    }

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set dataloader config
    interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
        },
        geom["geo"],
        {
            "dataset": "NamedArrayDataset",
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
            "num_workers": 1,
            "batch_size": cfg.TRAIN.batch_size.arm_interior,
        },
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        name="INTERIOR",
    )
    constraint = {interior_constraint.name: interior_constraint}

    # set validator
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))  # 0.5769
    MU = cfg.E / (2 * (1 + cfg.NU))  # 0.3846
    geom_validator = ppsci.validate.GeometryValidator(
        {
            "lambda_": lambda out: out["lambda_"],
            "mu": lambda out: out["mu"],
        },
        {
            "lambda_": LAMBDA_,
            "mu": MU,
        },
        geom["geo"],
        {
            "dataset": "NamedArrayDataset",
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
            "total_size": cfg.EVAL.total_size.validator,
            "batch_size": cfg.EVAL.batch_size.validator,
        },
        ppsci.loss.MSELoss("sum"),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="geo_eval",
    )
    validator = {geom_validator.name: geom_validator}

    # set visualizer(optional)
    # add inferencer data
    samples = geom["geo"].sample_interior(
        cfg.TRAIN.batch_size.visualizer_vtu,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict = {
        k: v for k, v in samples.items() if k in cfg.MODEL.disp_net.input_keys
    }
    visualizer = {
        "visulzie_lambda_mu": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "lambda": lambda out: out["lambda_"],
                "mu": lambda out: out["mu"],
            },
            prefix="vis",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        eval_freq=cfg.TRAIN.eval_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_with_no_grad=cfg.TRAIN.eval_with_no_grad,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
    )

    # train model
    solver.train()

    # plot losses
    solver.plot_loss_history(by_epoch=False, smooth_step=1, use_semilogy=True)


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    inverse_lambda_net = ppsci.arch.MLP(**cfg.MODEL.inverse_lambda_net)
    inverse_mu_net = ppsci.arch.MLP(**cfg.MODEL.inverse_mu_net)
    # wrap to a model_list
    model = ppsci.arch.ModelList(
        (disp_net, stress_net, inverse_lambda_net, inverse_mu_net)
    )

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set validator
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))  # 0.57692
    MU = cfg.E / (2 * (1 + cfg.NU))  # 0.38462
    geom_validator = ppsci.validate.GeometryValidator(
        {
            "lambda_": lambda out: out["lambda_"],
            "mu": lambda out: out["mu"],
        },
        {
            "lambda_": LAMBDA_,
            "mu": MU,
        },
        geom["geo"],
        {
            "dataset": "NamedArrayDataset",
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
            "total_size": cfg.EVAL.total_size.validator,
            "batch_size": cfg.EVAL.batch_size.validator,
        },
        ppsci.loss.MSELoss("sum"),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="geo_eval",
    )
    validator = {geom_validator.name: geom_validator}

    # set visualizer(optional)
    # add inferencer data
    samples = geom["geo"].sample_interior(
        cfg.EVAL.batch_size.visualizer_vtu,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict = {
        k: v for k, v in samples.items() if k in cfg.MODEL.disp_net.input_keys
    }
    visualizer = {
        "visulzie_lambda_mu": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "lambda": lambda out: out["lambda_"],
                "mu": lambda out: out["mu"],
            },
            prefix="vis",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        log_freq=cfg.log_freq,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


@hydra.main(
    version_base=None, config_path="./conf", config_name="inverse_parameter.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

from os import path as osp

import hydra
import numpy as np
from omegaconf import DictConfig
from paddle import distributed as dist

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")
    # set parallel
    enable_parallel = dist.get_world_size() > 1

    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    # wrap to a model_list
    model_list = ppsci.arch.ModelList((disp_net, stress_net))

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model_list)

    # specify parameters
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))
    MU = cfg.E / (2 * (1 + cfg.NU))

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
        )
    }

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    geom = {"geo": control_arm}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    arm_left_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": cfg.T[0], "traction_y": cfg.T[1], "traction_z": cfg.T[2]},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.arm_left},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - cfg.CIRCLE_LEFT_CENTER_XY[0])
            + np.square(y - cfg.CIRCLE_LEFT_CENTER_XY[1])
        )
        <= cfg.CIRCLE_LEFT_RADIUS + 1e-1,
        name="BC_LEFT",
    )
    arm_right_constraint = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.arm_right},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - cfg.CIRCLE_RIGHT_CENTER_XZ[0])
            + np.square(z - cfg.CIRCLE_RIGHT_CENTER_XZ[1])
        )
        <= cfg.CIRCLE_RIGHT_RADIUS + 1e-1,
        weight_dict=cfg.TRAIN.weight.arm_right,
        name="BC_RIGHT",
    )
    arm_surface_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.arm_surface},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - cfg.CIRCLE_LEFT_CENTER_XY[0])
            + np.square(y - cfg.CIRCLE_LEFT_CENTER_XY[1])
        )
        > cfg.CIRCLE_LEFT_RADIUS + 1e-1,
        name="BC_SURFACE",
    )
    arm_interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "equilibrium_z": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.arm_interior},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "equilibrium_z": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_zz": "sdf",
            "stress_disp_xy": "sdf",
            "stress_disp_xz": "sdf",
            "stress_disp_yz": "sdf",
        },
        name="INTERIOR",
    )

    # re-assign to cfg.TRAIN.iters_per_epoch
    if enable_parallel:
        cfg.TRAIN.iters_per_epoch = len(arm_left_constraint.data_loader)

    # wrap constraints togetherg
    constraint = {
        arm_left_constraint.name: arm_left_constraint,
        arm_right_constraint.name: arm_right_constraint,
        arm_surface_constraint.name: arm_surface_constraint,
        arm_interior_constraint.name: arm_interior_constraint,
    }

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
        "visulzie_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
                "sigma_xx": lambda out: out["sigma_xx"],
                "sigma_yy": lambda out: out["sigma_yy"],
                "sigma_zz": lambda out: out["sigma_zz"],
                "sigma_xy": lambda out: out["sigma_xy"],
                "sigma_xz": lambda out: out["sigma_xz"],
                "sigma_yz": lambda out: out["sigma_yz"],
            },
            prefix="vis",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model_list,
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
        visualizer=visualizer,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # train model
    solver.train()

    # plot losses
    solver.plot_loss_history(by_epoch=True, smooth_step=1)


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    # wrap to a model_list
    model_list = ppsci.arch.ModelList((disp_net, stress_net))

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

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
        "visulzie_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
                "sigma_xx": lambda out: out["sigma_xx"],
                "sigma_yy": lambda out: out["sigma_yy"],
                "sigma_zz": lambda out: out["sigma_zz"],
                "sigma_xy": lambda out: out["sigma_xy"],
                "sigma_xz": lambda out: out["sigma_xz"],
                "sigma_yz": lambda out: out["sigma_yz"],
            },
            prefix="vis",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model_list,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        geom=geom,
        log_freq=cfg.log_freq,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )

    # visualize prediction after finished training
    solver.visualize()


def export(cfg: DictConfig):
    from paddle.static import InputSpec

    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    # wrap to a model_list
    model_list = ppsci.arch.ModelList((disp_net, stress_net))

    # load pretrained model
    solver = ppsci.solver.Solver(
        model=model_list, pretrained_model_path=cfg.INFER.pretrained_model_path
    )

    # export models
    input_spec = [
        {
            key: InputSpec([None, 1], "float32", name=key)
            for key in cfg.MODEL.disp_net.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor
    from ppsci.visualize import vtu

    # set model predictor
    predictor = pinn_predictor.PINNPredictor(cfg)

    # set geometry
    control_arm = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

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

    output_dict = predictor.predict(pred_input_dict, cfg.INFER.batch_size)

    # mapping data to output_keys
    output_keys = cfg.MODEL.disp_net.output_keys + cfg.MODEL.stress_net.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(output_keys, output_dict.keys())
    }
    output_dict.update(pred_input_dict)

    vtu.save_vtu_from_dict(
        osp.join(cfg.output_dir, "vis"),
        output_dict,
        cfg.MODEL.disp_net.input_keys,
        output_keys,
        1,
    )


@hydra.main(
    version_base=None, config_path="./conf", config_name="forward_analysis.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

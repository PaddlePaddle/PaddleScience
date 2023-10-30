"""
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html
STL data files download link: https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
pretrained model download link: https://paddle-org.bj.bcebos.com/paddlescience/models/aneurysm/aneurysm_pretrained.pdparams
"""

from os import path as osp

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger
from ppsci.utils import reader


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL.model)

    # set equation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            cfg.NU * cfg.SCALE, cfg.RHO, 3, False
        ),
        "NormalDotVec": ppsci.equation.NormalDotVec(("u", "v", "w")),
    }

    # set geometry
    inlet_geo = ppsci.geometry.Mesh(cfg.INLET_PATH)
    outlet_geo = ppsci.geometry.Mesh(cfg.OUTLET_PATH)
    noslip_geo = ppsci.geometry.Mesh(cfg.NOSLIP_PATH)
    integral_geo = ppsci.geometry.Mesh(cfg.INTEGRAL_PATH)
    interior_geo = ppsci.geometry.Mesh(cfg.CLOSED_PATH)

    # inlet velocity profile
    CENTER = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    SCALE = 0.4

    # normalize meshes
    inlet_geo = inlet_geo.translate(-np.array(CENTER)).scale(SCALE)
    outlet_geo = outlet_geo.translate(-np.array(CENTER)).scale(SCALE)
    noslip_geo = noslip_geo.translate(-np.array(CENTER)).scale(SCALE)
    integral_geo = integral_geo.translate(-np.array(CENTER)).scale(SCALE)
    interior_geo = interior_geo.translate(-np.array(CENTER)).scale(SCALE)
    geom = {
        "inlet_geo": inlet_geo,
        "outlet_geo": outlet_geo,
        "noslip_geo": noslip_geo,
        "integral_geo": integral_geo,
        "interior_geo": interior_geo,
    }

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
    INLET_NORMAL = (0.8526, -0.428, 0.299)
    INLET_AREA = 21.1284 * (SCALE**2)
    INLET_CENTER = (-4.24298030045776, 4.082857101816247, -4.637790193399717)
    INLET_RADIUS = np.sqrt(INLET_AREA / np.pi)
    INLET_VEL = 1.5

    def _compute_parabola(_in):
        centered_x = _in["x"] - INLET_CENTER[0]
        centered_y = _in["y"] - INLET_CENTER[1]
        centered_z = _in["z"] - INLET_CENTER[2]
        distance = np.sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = INLET_VEL * np.maximum((1 - (distance / INLET_RADIUS) ** 2), 0)
        return parabola

    def inlet_u_ref_func(_in):
        return INLET_NORMAL[0] * _compute_parabola(_in)

    def inlet_v_ref_func(_in):
        return INLET_NORMAL[1] * _compute_parabola(_in)

    def inlet_w_ref_func(_in):
        return INLET_NORMAL[2] * _compute_parabola(_in)

    bc_inlet = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": inlet_u_ref_func, "v": inlet_v_ref_func, "w": inlet_w_ref_func},
        geom["inlet_geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_inlet},
        ppsci.loss.MSELoss("sum"),
        name="inlet",
    )
    bc_outlet = ppsci.constraint.BoundaryConstraint(
        {"p": lambda d: d["p"]},
        {"p": 0},
        geom["outlet_geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_outlet},
        ppsci.loss.MSELoss("sum"),
        name="outlet",
    )
    bc_noslip = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["noslip_geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_noslip},
        ppsci.loss.MSELoss("sum"),
        name="no_slip",
    )
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom["interior_geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.pde_constraint},
        ppsci.loss.MSELoss("sum"),
        name="interior",
    )
    igc_outlet = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vec": 2.54},
        geom["outlet_geo"],
        {
            **train_dataloader_cfg,
            "iters_per_epoch": cfg.TRAIN.iters_igc_outlet,
            "batch_size": cfg.TRAIN.batch_size.igc_outlet,
            "integral_batch_size": cfg.TRAIN.integral_batch_size.igc_outlet,
        },
        ppsci.loss.IntegralLoss("sum"),
        weight_dict=cfg.TRAIN.weight.igc_outlet,
        name="igc_outlet",
    )
    igc_integral = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vec": -2.54},
        geom["integral_geo"],
        {
            **train_dataloader_cfg,
            "iters_per_epoch": cfg.TRAIN.iters_igc_integral,
            "batch_size": cfg.TRAIN.batch_size.igc_integral,
            "integral_batch_size": cfg.TRAIN.integral_batch_size.igc_integral,
        },
        ppsci.loss.IntegralLoss("sum"),
        weight_dict=cfg.TRAIN.weight.igc_integral,
        name="igc_integral",
    )
    # wrap constraints together
    constraint = {
        bc_inlet.name: bc_inlet,
        bc_outlet.name: bc_outlet,
        bc_noslip.name: bc_noslip,
        pde_constraint.name: pde_constraint,
        igc_outlet.name: igc_outlet,
        igc_integral.name: igc_integral,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        cfg.TRAIN.learning_rate,
        cfg.TRAIN.gamma,
        cfg.TRAIN.epochs * cfg.TRAIN.iters_per_epoch // 100,
        by_epoch=False,
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    eval_data_dict = reader.load_csv_file(
        "./data/aneurysm_parabolicInlet_sol0.csv",
        ("x", "y", "z", "u", "v", "w", "p"),
        {
            "x": "Points:0",
            "y": "Points:1",
            "z": "Points:2",
            "u": "U:0",
            "v": "U:1",
            "w": "U:2",
            "p": "p",
        },
    )
    input_dict = {
        "x": (eval_data_dict["x"] - CENTER[0]) * SCALE,
        "y": (eval_data_dict["y"] - CENTER[1]) * SCALE,
        "z": (eval_data_dict["z"] - CENTER[2]) * SCALE,
    }
    if "area" in input_dict.keys():
        input_dict["area"] *= SCALE ** (equation["NavierStokes"].dim)

    label_dict = {
        "p": eval_data_dict["p"],
        "u": eval_data_dict["u"],
        "v": eval_data_dict["v"],
        "w": eval_data_dict["w"],
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
        },
        "sampler": {"name": "BatchSampler"},
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size.sup_validator},
        ppsci.loss.MSELoss("mean"),
        {
            "p": lambda out: out["p"],
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="ref_u_v_w_p",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visulzie_u_v_w_p": ppsci.visualize.VisualizerVtu(
            input_dict,
            {
                "p": lambda out: out["p"],
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            batch_size=cfg.EVAL.batch_size.sup_validator,
            prefix="result_u_v_w_p",
        ),
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
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        log_freq=cfg.TRAIN.log_freq,
        eval_freq=cfg.TRAIN.eval_freq,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL.model)

    # set equation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            cfg.NU * cfg.SCALE, cfg.RHO, 3, False
        ),
        "NormalDotVec": ppsci.equation.NormalDotVec(("u", "v", "w")),
    }

    # inlet velocity profile
    CENTER = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    SCALE = 0.4

    # set validator
    eval_data_dict = reader.load_csv_file(
        "./data/aneurysm_parabolicInlet_sol0.csv",
        ("x", "y", "z", "u", "v", "w", "p"),
        {
            "x": "Points:0",
            "y": "Points:1",
            "z": "Points:2",
            "u": "U:0",
            "v": "U:1",
            "w": "U:2",
            "p": "p",
        },
    )
    input_dict = {
        "x": (eval_data_dict["x"] - CENTER[0]) * SCALE,
        "y": (eval_data_dict["y"] - CENTER[1]) * SCALE,
        "z": (eval_data_dict["z"] - CENTER[2]) * SCALE,
    }
    if "area" in input_dict.keys():
        input_dict["area"] *= SCALE ** (equation["NavierStokes"].dim)

    label_dict = {
        "p": eval_data_dict["p"],
        "u": eval_data_dict["u"],
        "v": eval_data_dict["v"],
        "w": eval_data_dict["w"],
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
        },
        "sampler": {"name": "BatchSampler"},
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size.sup_validator},
        ppsci.loss.MSELoss("mean"),
        {
            "p": lambda out: out["p"],
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="ref_u_v_w_p",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visulzie_u_v_w_p": ppsci.visualize.VisualizerVtu(
            input_dict,
            {
                "p": lambda out: out["p"],
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            batch_size=cfg.EVAL.batch_size.sup_validator,
            prefix="result_u_v_w_p",
        ),
    }

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()


@hydra.main(version_base=None, config_path="./conf", config_name="aneurysm.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

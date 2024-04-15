"""
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/foundational/linear_elasticity.html
STL data files download link: https://paddle-org.bj.bcebos.com/paddlescience/datasets/bracket/bracket_dataset.tar
pretrained model download link: https://paddle-org.bj.bcebos.com/paddlescience/models/bracket/bracket_pretrained.pdparams
"""

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci


def train(cfg: DictConfig):
    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    # wrap to a model_list
    model = ppsci.arch.ModelList((disp_net, stress_net))

    # specify parameters
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))
    MU = cfg.E / (2 * (1 + cfg.NU))
    MU_C = 0.01 * MU
    LAMBDA_ = LAMBDA_ / MU_C
    MU = MU / MU_C
    SIGMA_NORMALIZATION = cfg.CHARACTERISTIC_LENGTH / (
        cfg.CHARACTERISTIC_DISPLACEMENT * MU_C
    )
    T = -4.0e4 * SIGMA_NORMALIZATION

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            lambda_=LAMBDA_, mu=MU, dim=3
        )
    }

    # set geometry
    support = ppsci.geometry.Mesh(cfg.SUPPORT_PATH)
    bracket = ppsci.geometry.Mesh(cfg.BRACKET_PATH)
    aux_lower = ppsci.geometry.Mesh(cfg.AUX_LOWER_PATH)
    aux_upper = ppsci.geometry.Mesh(cfg.AUX_UPPER_PATH)
    cylinder_hole = ppsci.geometry.Mesh(cfg.CYLINDER_HOLE_PATH)
    cylinder_lower = ppsci.geometry.Mesh(cfg.CYLINDER_LOWER_PATH)
    cylinder_upper = ppsci.geometry.Mesh(cfg.CYLINDER_UPPER_PATH)
    # geometry bool operation
    curve_lower = aux_lower - cylinder_lower
    curve_upper = aux_upper - cylinder_upper
    geo = support + bracket + curve_lower + curve_upper - cylinder_hole
    geom = {"geo": geo}

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
    SUPPORT_ORIGIN = (-1, -1, -1)
    BRACKET_ORIGIN = (-0.75, -1, -0.1)
    BRACKET_DIM = (1.75, 2, 0.2)
    BOUNDS_SUPPORT_X = (-1, -0.65)
    BOUNDS_SUPPORT_Y = (-1, 1)
    BOUNDS_SUPPORT_Z = (-1, 1)
    BOUNDS_BRACKET_X = (-0.65, 1)
    BOUNDS_BRACKET_Y = (-1, 1)
    BOUNDS_BRACKET_Z = (-0.1, 0.1)

    bc_back = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_back},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: x == SUPPORT_ORIGIN[0],
        weight_dict=cfg.TRAIN.weight.bc_back,
        name="BC_BACK",
    )
    bc_front = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": T},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_front},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: x == BRACKET_ORIGIN[0] + BRACKET_DIM[0],
        name="BC_FRONT",
    )
    bc_surface = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_surface},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.logical_and(
            x > SUPPORT_ORIGIN[0] + 1e-7, x < BRACKET_ORIGIN[0] + BRACKET_DIM[0] - 1e-7
        ),
        name="BC_SURFACE",
    )
    support_interior = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "equilibrium_z": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.support_interior},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_SUPPORT_X[0] < x)
            & (x < BOUNDS_SUPPORT_X[1])
            & (BOUNDS_SUPPORT_Y[0] < y)
            & (y < BOUNDS_SUPPORT_Y[1])
            & (BOUNDS_SUPPORT_Z[0] < z)
            & (z < BOUNDS_SUPPORT_Z[1])
        ),
        weight_dict={
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_zz": "sdf",
            "stress_disp_xy": "sdf",
            "stress_disp_xz": "sdf",
            "stress_disp_yz": "sdf",
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "equilibrium_z": "sdf",
        },
        name="SUPPORT_INTERIOR",
    )
    bracket_interior = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "equilibrium_z": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bracket_interior},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_BRACKET_X[0] < x)
            & (x < BOUNDS_BRACKET_X[1])
            & (BOUNDS_BRACKET_Y[0] < y)
            & (y < BOUNDS_BRACKET_Y[1])
            & (BOUNDS_BRACKET_Z[0] < z)
            & (z < BOUNDS_BRACKET_Z[1])
        ),
        weight_dict={
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_zz": "sdf",
            "stress_disp_xy": "sdf",
            "stress_disp_xz": "sdf",
            "stress_disp_yz": "sdf",
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "equilibrium_z": "sdf",
        },
        name="BRACKET_INTERIOR",
    )
    # wrap constraints together
    constraint = {
        bc_back.name: bc_back,
        bc_front.name: bc_front,
        bc_surface.name: bc_surface,
        support_interior.name: support_interior,
        bracket_interior.name: bracket_interior,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    ref_xyzu = ppsci.utils.reader.load_csv_file(
        cfg.DEFORMATION_X_PATH,
        ("x", "y", "z", "u"),
        {
            "x": "X Location (m)",
            "y": "Y Location (m)",
            "z": "Z Location (m)",
            "u": "Directional Deformation (m)",
        },
        "\t",
    )
    ref_v = ppsci.utils.reader.load_csv_file(
        cfg.DEFORMATION_Y_PATH,
        ("v",),
        {"v": "Directional Deformation (m)"},
        "\t",
    )
    ref_w = ppsci.utils.reader.load_csv_file(
        cfg.DEFORMATION_Z_PATH,
        ("w",),
        {"w": "Directional Deformation (m)"},
        "\t",
    )

    ref_sxx = ppsci.utils.reader.load_csv_file(
        cfg.NORMAL_X_PATH,
        ("sigma_xx",),
        {"sigma_xx": "Normal Stress (Pa)"},
        "\t",
    )
    ref_syy = ppsci.utils.reader.load_csv_file(
        cfg.NORMAL_Y_PATH,
        ("sigma_yy",),
        {"sigma_yy": "Normal Stress (Pa)"},
        "\t",
    )
    ref_szz = ppsci.utils.reader.load_csv_file(
        cfg.NORMAL_Z_PATH,
        ("sigma_zz",),
        {"sigma_zz": "Normal Stress (Pa)"},
        "\t",
    )

    ref_sxy = ppsci.utils.reader.load_csv_file(
        cfg.SHEAR_XY_PATH,
        ("sigma_xy",),
        {"sigma_xy": "Shear Stress (Pa)"},
        "\t",
    )
    ref_sxz = ppsci.utils.reader.load_csv_file(
        cfg.SHEAR_XZ_PATH,
        ("sigma_xz",),
        {"sigma_xz": "Shear Stress (Pa)"},
        "\t",
    )
    ref_syz = ppsci.utils.reader.load_csv_file(
        cfg.SHEAR_YZ_PATH,
        ("sigma_yz",),
        {"sigma_yz": "Shear Stress (Pa)"},
        "\t",
    )

    input_dict = {
        "x": ref_xyzu["x"],
        "y": ref_xyzu["y"],
        "z": ref_xyzu["z"],
    }
    label_dict = {
        "u": ref_xyzu["u"] / cfg.CHARACTERISTIC_DISPLACEMENT,
        "v": ref_v["v"] / cfg.CHARACTERISTIC_DISPLACEMENT,
        "w": ref_w["w"] / cfg.CHARACTERISTIC_DISPLACEMENT,
        "sigma_xx": ref_sxx["sigma_xx"] * SIGMA_NORMALIZATION,
        "sigma_yy": ref_syy["sigma_yy"] * SIGMA_NORMALIZATION,
        "sigma_zz": ref_szz["sigma_zz"] * SIGMA_NORMALIZATION,
        "sigma_xy": ref_sxy["sigma_xy"] * SIGMA_NORMALIZATION,
        "sigma_xz": ref_sxz["sigma_xz"] * SIGMA_NORMALIZATION,
        "sigma_yz": ref_syz["sigma_yz"] * SIGMA_NORMALIZATION,
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size.sup_validator},
        ppsci.loss.MSELoss("mean"),
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
        metric={"MSE": ppsci.metric.MSE()},
        name="commercial_ref_u_v_w_sigmas",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visualize_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            input_dict,
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
            prefix="result_u_v_w_sigmas",
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
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
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
    # set model
    disp_net = ppsci.arch.MLP(**cfg.MODEL.disp_net)
    stress_net = ppsci.arch.MLP(**cfg.MODEL.stress_net)
    # wrap to a model_list
    model = ppsci.arch.ModelList((disp_net, stress_net))

    # Specify parameters
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))
    MU = cfg.E / (2 * (1 + cfg.NU))
    MU_C = 0.01 * MU
    LAMBDA_ = LAMBDA_ / MU_C
    MU = MU / MU_C
    SIGMA_NORMALIZATION = cfg.CHARACTERISTIC_LENGTH / (
        cfg.CHARACTERISTIC_DISPLACEMENT * MU_C
    )

    # set validator
    ref_xyzu = ppsci.utils.reader.load_csv_file(
        cfg.DEFORMATION_X_PATH,
        ("x", "y", "z", "u"),
        {
            "x": "X Location (m)",
            "y": "Y Location (m)",
            "z": "Z Location (m)",
            "u": "Directional Deformation (m)",
        },
        "\t",
    )
    ref_v = ppsci.utils.reader.load_csv_file(
        cfg.DEFORMATION_Y_PATH,
        ("v",),
        {"v": "Directional Deformation (m)"},
        "\t",
    )
    ref_w = ppsci.utils.reader.load_csv_file(
        cfg.DEFORMATION_Z_PATH,
        ("w",),
        {"w": "Directional Deformation (m)"},
        "\t",
    )

    ref_sxx = ppsci.utils.reader.load_csv_file(
        cfg.NORMAL_X_PATH,
        ("sigma_xx",),
        {"sigma_xx": "Normal Stress (Pa)"},
        "\t",
    )
    ref_syy = ppsci.utils.reader.load_csv_file(
        cfg.NORMAL_Y_PATH,
        ("sigma_yy",),
        {"sigma_yy": "Normal Stress (Pa)"},
        "\t",
    )
    ref_szz = ppsci.utils.reader.load_csv_file(
        cfg.NORMAL_Z_PATH,
        ("sigma_zz",),
        {"sigma_zz": "Normal Stress (Pa)"},
        "\t",
    )

    ref_sxy = ppsci.utils.reader.load_csv_file(
        cfg.SHEAR_XY_PATH,
        ("sigma_xy",),
        {"sigma_xy": "Shear Stress (Pa)"},
        "\t",
    )
    ref_sxz = ppsci.utils.reader.load_csv_file(
        cfg.SHEAR_XZ_PATH,
        ("sigma_xz",),
        {"sigma_xz": "Shear Stress (Pa)"},
        "\t",
    )
    ref_syz = ppsci.utils.reader.load_csv_file(
        cfg.SHEAR_YZ_PATH,
        ("sigma_yz",),
        {"sigma_yz": "Shear Stress (Pa)"},
        "\t",
    )

    input_dict = {
        "x": ref_xyzu["x"],
        "y": ref_xyzu["y"],
        "z": ref_xyzu["z"],
    }
    label_dict = {
        "u": ref_xyzu["u"] / cfg.CHARACTERISTIC_DISPLACEMENT,
        "v": ref_v["v"] / cfg.CHARACTERISTIC_DISPLACEMENT,
        "w": ref_w["w"] / cfg.CHARACTERISTIC_DISPLACEMENT,
        "sigma_xx": ref_sxx["sigma_xx"] * SIGMA_NORMALIZATION,
        "sigma_yy": ref_syy["sigma_yy"] * SIGMA_NORMALIZATION,
        "sigma_zz": ref_szz["sigma_zz"] * SIGMA_NORMALIZATION,
        "sigma_xy": ref_sxy["sigma_xy"] * SIGMA_NORMALIZATION,
        "sigma_xz": ref_sxz["sigma_xz"] * SIGMA_NORMALIZATION,
        "sigma_yz": ref_syz["sigma_yz"] * SIGMA_NORMALIZATION,
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size.sup_validator},
        ppsci.loss.MSELoss("mean"),
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
        metric={"MSE": ppsci.metric.MSE()},
        name="commercial_ref_u_v_w_sigmas",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visualize_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            input_dict,
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
            prefix="result_u_v_w_sigmas",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate
    solver.eval()
    # visualize prediction
    solver.visualize()


@hydra.main(version_base=None, config_path="./conf", config_name="bracket.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

"""
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html
"""

import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import reader


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            cfg.NU * cfg.SCALE, cfg.RHO, cfg.DIM, False
        ),
        "NormalDotVec": ppsci.equation.NormalDotVec(("u", "v", "w")),
    }

    # set geometry
    inlet_geo = ppsci.geometry.Mesh(cfg.INLET_STL_PATH)
    outlet_geo = ppsci.geometry.Mesh(cfg.OUTLET_STL_PATH)
    noslip_geo = ppsci.geometry.Mesh(cfg.NOSLIP_STL_PATH)
    integral_geo = ppsci.geometry.Mesh(cfg.INTEGRAL_STL_PATH)
    interior_geo = ppsci.geometry.Mesh(cfg.INTERIOR_STL_PATH)

    # normalize meshes
    inlet_geo = inlet_geo.translate(-np.array(cfg.CENTER)).scale(cfg.SCALE)
    outlet_geo = outlet_geo.translate(-np.array(cfg.CENTER)).scale(cfg.SCALE)
    noslip_geo = noslip_geo.translate(-np.array(cfg.CENTER)).scale(cfg.SCALE)
    integral_geo = integral_geo.translate(-np.array(cfg.CENTER)).scale(cfg.SCALE)
    interior_geo = interior_geo.translate(-np.array(cfg.CENTER)).scale(cfg.SCALE)
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
    INLET_AREA = 21.1284 * (cfg.SCALE**2)
    INLET_RADIUS = np.sqrt(INLET_AREA / np.pi)

    def _compute_parabola(_in):
        centered_x = _in["x"] - cfg.INLET_CENTER[0]
        centered_y = _in["y"] - cfg.INLET_CENTER[1]
        centered_z = _in["z"] - cfg.INLET_CENTER[2]
        distance = np.sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = cfg.INLET_VEL * np.maximum((1 - (distance / INLET_RADIUS) ** 2), 0)
        return parabola

    def inlet_u_ref_func(_in):
        return cfg.INLET_NORMAL[0] * _compute_parabola(_in)

    def inlet_v_ref_func(_in):
        return cfg.INLET_NORMAL[1] * _compute_parabola(_in)

    def inlet_w_ref_func(_in):
        return cfg.INLET_NORMAL[2] * _compute_parabola(_in)

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
    pde = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom["interior_geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.pde},
        ppsci.loss.MSELoss("sum"),
        name="interior",
    )
    igc_outlet = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vec": 2.54},
        geom["outlet_geo"],
        {
            **train_dataloader_cfg,
            "iters_per_epoch": cfg.TRAIN.iters_integral.igc_outlet,
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
            "iters_per_epoch": cfg.TRAIN.iters_integral.igc_integral,
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
        pde.name: pde,
        igc_outlet.name: igc_outlet,
        igc_integral.name: igc_integral,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # set validator
    eval_data_dict = reader.load_csv_file(
        cfg.EVAL_CSV_PATH,
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
        "x": (eval_data_dict["x"] - cfg.CENTER[0]) * cfg.SCALE,
        "y": (eval_data_dict["y"] - cfg.CENTER[1]) * cfg.SCALE,
        "z": (eval_data_dict["z"] - cfg.CENTER[2]) * cfg.SCALE,
    }
    if "area" in input_dict.keys():
        input_dict["area"] *= cfg.SCALE ** (equation["NavierStokes"].dim)

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
        "visualize_u_v_w_p": ppsci.visualize.VisualizerVtu(
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
        log_freq=cfg.log_freq,
        eval_during_train=True,
        eval_freq=cfg.TRAIN.eval_freq,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
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
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set validator
    eval_data_dict = reader.load_csv_file(
        cfg.EVAL_CSV_PATH,
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
        "x": (eval_data_dict["x"] - cfg.CENTER[0]) * cfg.SCALE,
        "y": (eval_data_dict["y"] - cfg.CENTER[1]) * cfg.SCALE,
        "z": (eval_data_dict["z"] - cfg.CENTER[2]) * cfg.SCALE,
    }

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

    # set visualizer
    visualizer = {
        "visualize_u_v_w_p": ppsci.visualize.VisualizerVtu(
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


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)
    eval_data_dict = reader.load_csv_file(
        cfg.EVAL_CSV_PATH,
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
        "x": (eval_data_dict["x"] - cfg.CENTER[0]) * cfg.SCALE,
        "y": (eval_data_dict["y"] - cfg.CENTER[1]) * cfg.SCALE,
        "z": (eval_data_dict["z"] - cfg.CENTER[2]) * cfg.SCALE,
    }
    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    ppsci.visualize.save_vtu_from_dict(
        "./aneurysm_pred.vtu",
        {**input_dict, **output_dict},
        input_dict.keys(),
        cfg.MODEL.output_keys,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="aneurysm.yaml")
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

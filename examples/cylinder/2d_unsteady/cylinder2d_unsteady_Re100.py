# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import hydra
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger
from ppsci.utils import reader


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(cfg.VISCOSITY, cfg.DENSITY, 2, True)
    }

    # set timestamps
    train_timestamps = np.linspace(
        cfg.TIME_START, cfg.TIME_END, cfg.NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")
    train_timestamps = np.random.choice(train_timestamps, cfg.TRAIN_NUM_TIMESTAMPS)
    train_timestamps.sort()
    t0 = np.array([cfg.TIME_START], dtype="float32")

    val_timestamps = np.linspace(
        cfg.TIME_START, cfg.TIME_END, cfg.NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")

    logger.message(f"train_timestamps: {train_timestamps.tolist()}")
    logger.message(f"val_timestamps: {val_timestamps.tolist()}")

    # set time-geometry
    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(
                cfg.TIME_START,
                cfg.TIME_END,
                timestamps=np.concatenate((t0, train_timestamps), axis=0),
            ),
            ppsci.geometry.PointCloud(
                reader.load_csv_file(
                    cfg.DOMAIN_TRAIN_PATH,
                    ("x", "y"),
                    alias_dict={"x": "Points:0", "y": "Points:1"},
                ),
                ("x", "y"),
            ),
        ),
        "time_rect_eval": ppsci.geometry.PointCloud(
            reader.load_csv_file(
                cfg.DOMAIN_EVAL_PATH,
                ("t", "x", "y"),
            ),
            ("t", "x", "y"),
        ),
    }

    # pde/bc/sup constraint use t1~tn, initial constraint use t0
    NTIME_PDE = len(train_timestamps)
    ALIAS_DICT = {"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"}

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "batch_size": cfg.NPOINT_PDE * NTIME_PDE,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )
    bc_inlet_cylinder = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": cfg.DOMAIN_INLET_CYLINDER_PATH,
                "input_keys": ("x", "y"),
                "label_keys": ("u", "v"),
                "alias_dict": ALIAS_DICT,
                "weight_dict": {"u": 10, "v": 10},
                "timestamps": train_timestamps,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="BC_inlet_cylinder",
    )
    bc_outlet = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": cfg.DOMAIN_OUTLET_PATH,
                "input_keys": ("x", "y"),
                "label_keys": ("p",),
                "alias_dict": ALIAS_DICT,
                "timestamps": train_timestamps,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="BC_outlet",
    )
    ic = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": cfg.IC0_1_PATH,
                "input_keys": ("x", "y"),
                "label_keys": ("u", "v", "p"),
                "alias_dict": ALIAS_DICT,
                "weight_dict": {"u": 10, "v": 10, "p": 10},
                "timestamps": t0,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="IC",
    )
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": cfg.PROBE1_50_PATH,
                "input_keys": ("t", "x", "y"),
                "label_keys": ("u", "v"),
                "alias_dict": ALIAS_DICT,
                "weight_dict": {"u": 10, "v": 10},
                "timestamps": train_timestamps,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="Sup",
    )

    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_inlet_cylinder.name: bc_inlet_cylinder,
        bc_outlet.name: bc_outlet,
        ic.name: ic,
        sup_constraint.name: sup_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    NPOINT_EVAL = (
        cfg.NPOINT_PDE + cfg.NPOINT_INLET_CYLINDER + cfg.NPOINT_OUTLET
    ) * cfg.NUM_TIMESTAMPS
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect_eval"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # set visualizer(optional)
    vis_points = geom["time_rect_eval"].sample_interior(
        (cfg.NPOINT_PDE + cfg.NPOINT_INLET_CYLINDER + cfg.NPOINT_OUTLET)
        * cfg.NUM_TIMESTAMPS,
        evenly=True,
    )
    visualizer = {
        "visualize_u_v_p": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            num_timestamps=cfg.NUM_TIMESTAMPS,
            prefix="result_u_v_p",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        equation=equation,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
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

    # set equation
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(cfg.VISCOSITY, cfg.DENSITY, 2, True)
    }

    # set timestamps
    val_timestamps = np.linspace(
        cfg.TIME_START, cfg.TIME_END, cfg.NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")

    logger.message(f"val_timestamps: {val_timestamps.tolist()}")

    # set time-geometry
    geom = {
        "time_rect_eval": ppsci.geometry.PointCloud(
            reader.load_csv_file(
                cfg.DOMAIN_EVAL_PATH,
                ("t", "x", "y"),
            ),
            ("t", "x", "y"),
        ),
    }

    # set validator
    NPOINT_EVAL = (
        cfg.NPOINT_PDE + cfg.NPOINT_INLET_CYLINDER + cfg.NPOINT_OUTLET
    ) * cfg.NUM_TIMESTAMPS
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect_eval"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # set visualizer(optional)
    vis_points = geom["time_rect_eval"].sample_interior(
        (cfg.NPOINT_PDE + cfg.NPOINT_INLET_CYLINDER + cfg.NPOINT_OUTLET)
        * cfg.NUM_TIMESTAMPS,
        evenly=True,
    )
    visualizer = {
        "visualize_u_v_p": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            num_timestamps=cfg.NUM_TIMESTAMPS,
            prefix="result_u_v_p",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
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
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)
    # set time-geometry
    geom = {
        "time_rect_eval": ppsci.geometry.PointCloud(
            reader.load_csv_file(
                cfg.DOMAIN_EVAL_PATH,
                ("t", "x", "y"),
            ),
            ("t", "x", "y"),
        ),
    }
    NPOINT_EVAL = (
        cfg.NPOINT_PDE + cfg.NPOINT_INLET_CYLINDER + cfg.NPOINT_OUTLET
    ) * cfg.NUM_TIMESTAMPS
    input_dict = geom["time_rect_eval"].sample_interior(NPOINT_EVAL, evenly=True)
    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    ppsci.visualize.save_vtu_from_dict(
        "./cylinder2d_unsteady_Re100_pred.vtu",
        {**input_dict, **output_dict},
        input_dict.keys(),
        cfg.MODEL.output_keys,
        cfg.NUM_TIMESTAMPS,
    )


@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="cylinder2d_unsteady_Re100.yaml",
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

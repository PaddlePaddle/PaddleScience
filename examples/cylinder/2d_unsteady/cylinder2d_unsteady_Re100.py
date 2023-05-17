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

import numpy as np
from paddle import fluid

import ppsci
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import reader

if __name__ == "__main__":
    fluid.core.set_prim_eager_enabled(True)

    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(42)
    # set output directory
    OUTPUT_DIR = (
        "./output_cylinder2d_unsteady" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(
        ("t", "x", "y"), ("u", "v", "p"), 5, 50, "tanh", False, False
    )
    # set equation
    equation = {"NavierStokes": ppsci.equation.NavierStokes(0.02, 1.0, 2, True)}

    # set timestamps
    TIME_START, TIME_END = 1, 50
    NUM_TIMESTAMPS = 50
    TRAIN_NUM_TIMESTAMPS = 30

    train_timestamps = np.linspace(
        TIME_START, TIME_END, NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")
    train_timestamps = np.random.choice(train_timestamps, TRAIN_NUM_TIMESTAMPS)
    train_timestamps.sort()
    t0 = np.array([TIME_START], dtype="float32")

    val_timestamps = np.linspace(
        TIME_START, TIME_END, NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")

    logger.info(f"train_timestamps: {train_timestamps.tolist()}")
    logger.info(f"val_timestamps: {val_timestamps.tolist()}")

    # set time-geometry
    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(
                TIME_START,
                TIME_END,
                timestamps=np.concatenate((t0, train_timestamps), axis=0),
            ),
            ppsci.geometry.PointCloud(
                reader.load_csv_file(
                    "./datasets/domain_train.csv",
                    ("x", "y"),
                    alias_dict={"x": "Points:0", "y": "Points:1"},
                ),
                ("x", "y"),
            ),
        ),
        "time_rect_eval": ppsci.geometry.PointCloud(
            reader.load_csv_file(
                "./datasets/domain_eval.csv",
                ("t", "x", "y"),
            ),
            ("t", "x", "y"),
        ),
    }

    # set dataloader config
    ITERS_PER_EPOCH = 1

    # pde/bc/sup constraint use t1~tn, initial constraint use t0
    NPOINT_PDE, ntime_pde = 9420, len(train_timestamps)
    NPOINT_INLET_CYLINDER = 161
    NPOINT_OUTLET = 81
    ALIAS_DICT = {"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"}

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "batch_size": NPOINT_PDE * ntime_pde,
            "iters_per_epoch": ITERS_PER_EPOCH,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )
    bc_inlet_cylinder = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": "./datasets/domain_inlet_cylinder.csv",
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
                "file_path": "./datasets/domain_outlet.csv",
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
                "file_path": "./datasets/initial/ic0.1.csv",
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
                "file_path": "./datasets/probe/probe1_50.csv",
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

    # set training hyper-parameters
    EPOCHS = 40000 if not args.epochs else args.epochs
    EVAL_FREQ = 400

    # set optimizer
    optimizer = ppsci.optimizer.Adam(0.001)((model,))

    # set validator
    NPOINT_EVAL = (NPOINT_PDE + NPOINT_INLET_CYLINDER + NPOINT_OUTLET) * NUM_TIMESTAMPS
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geom["time_rect_eval"],
        {
            "dataset": "NamedArrayDataset",
            "total_size": NPOINT_EVAL,
            "batch_size": 10240,
            "sampler": {"name": "BatchSampler"},
        },
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # set visualizer(optional)
    vis_points = geom["time_rect_eval"].sample_interior(
        (NPOINT_PDE + NPOINT_INLET_CYLINDER + NPOINT_OUTLET) * NUM_TIMESTAMPS
    )
    visualizer = {
        "visulzie_u": ppsci.visualize.VisualizerVtu(
            vis_points,
            {"u": lambda d: d["u"], "v": lambda d: d["v"], "p": lambda d: d["p"]},
            num_timestamps=NUM_TIMESTAMPS,
            prefix="result_u",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        eval_freq=EVAL_FREQ,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate model from pretrained_model_path(optional)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    solver.eval()
    # visualize prediction from pretrained_model_path(optional)
    solver.visualize()

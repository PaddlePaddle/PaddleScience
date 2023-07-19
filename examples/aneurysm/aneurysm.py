# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/intermediate/adding_stl_files.html
STL data files download link: https://paddle-org.bj.bcebos.com/paddlescience/datasets/aneurysm/aneurysm_dataset.tar
"""

import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import reader

if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    # set output directory
    OUTPUT_DIR = (
        f"./output_aneurysm_seed{SEED}" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set model
    model = ppsci.arch.MLP(
        ("x", "y", "z"), ("u", "v", "w", "p"), 6, 512, "silu", weight_norm=True
    )

    # set equation
    NU = 0.025
    SCALE = 0.4
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(NU * SCALE, 1.0, 3, False),
        "NormalDotVec": ppsci.equation.NormalDotVec(("u", "v", "w")),
    }

    # set geometry
    inlet_geo = ppsci.geometry.Mesh("./stl/aneurysm_inlet.stl")
    outlet_geo = ppsci.geometry.Mesh("./stl/aneurysm_outlet.stl")
    noslip_geo = ppsci.geometry.Mesh("./stl/aneurysm_noslip.stl")
    integral_geo = ppsci.geometry.Mesh("./stl/aneurysm_integral.stl")
    interior_geo = ppsci.geometry.Mesh("./stl/aneurysm_closed.stl")

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
    ITERS_PER_EPOCH = 1000
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
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

    def compute_parabola(_in):
        centered_x = _in["x"] - INLET_CENTER[0]
        centered_y = _in["y"] - INLET_CENTER[1]
        centered_z = _in["z"] - INLET_CENTER[2]
        distance = np.sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = INLET_VEL * np.maximum((1 - (distance / INLET_RADIUS) ** 2), 0)
        return parabola

    def inlet_u_ref_func(_in):
        return INLET_NORMAL[0] * compute_parabola(_in)

    def inlet_v_ref_func(_in):
        return INLET_NORMAL[1] * compute_parabola(_in)

    def inlet_w_ref_func(_in):
        return INLET_NORMAL[2] * compute_parabola(_in)

    bc_inlet = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": inlet_u_ref_func, "v": inlet_v_ref_func, "w": inlet_w_ref_func},
        geom["inlet_geo"],
        {**train_dataloader_cfg, "batch_size": 1100},
        ppsci.loss.MSELoss("sum"),
        name="inlet",
    )
    bc_outlet = ppsci.constraint.BoundaryConstraint(
        {"p": lambda d: d["p"]},
        {"p": 0},
        geom["outlet_geo"],
        {**train_dataloader_cfg, "batch_size": 650},
        ppsci.loss.MSELoss("sum"),
        name="outlet",
    )
    bc_noslip = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["noslip_geo"],
        {**train_dataloader_cfg, "batch_size": 5200},
        ppsci.loss.MSELoss("sum"),
        name="no_slip",
    )
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom["interior_geo"],
        {**train_dataloader_cfg, "batch_size": 6000},
        ppsci.loss.MSELoss("sum"),
        name="interior",
    )
    igc_outlet = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vel": 2.54},
        geom["outlet_geo"],
        {
            **train_dataloader_cfg,
            "iters_per_epoch": 100,
            "batch_size": 1,
            "integral_batch_size": 310,
        },
        ppsci.loss.IntegralLoss("sum"),
        weight_dict={"normal_dot_vel": 0.1},
        name="igc_outlet",
    )
    igc_integral = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vel": -2.54},
        geom["integral_geo"],
        {
            **train_dataloader_cfg,
            "iters_per_epoch": 100,
            "batch_size": 1,
            "integral_batch_size": 310,
        },
        ppsci.loss.IntegralLoss("sum"),
        weight_dict={"normal_dot_vel": 0.1},
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

    # set training hyper-parameters
    EPOCHS = 1500 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        0.95,
        EPOCHS * ITERS_PER_EPOCH // 100,
        by_epoch=False,
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

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
            "weight": {k: np.ones_like(v) for k, v in label_dict.items()},
        },
        "sampler": {"name": "BatchSampler"},
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": 4096},
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
            batch_size=4096,
            prefix="result_u_v_w_p",
        ),
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=20,
        eval_during_train=True,
        log_freq=20,
        eval_freq=20,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # directly evaluate pretrained model(optional)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/best_model",
        eval_with_no_grad=True,
    )
    solver.eval()
    # visualize prediction for pretrained model(optional)
    solver.visualize()

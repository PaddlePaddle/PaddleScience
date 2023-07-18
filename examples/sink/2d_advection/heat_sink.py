# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.s

"""
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/scalar_transport.html#advection-diffusion
"""

import ppsci
import paddle
import numpy as np
import os.path as osp
from ppsci.utils import config
from ppsci.utils import logger
from vtk.util.numpy_support import vtk_to_numpy

if __name__ == "__main__":
    # initialization
    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    OUTPUT_DIR = (
        "./output" if not args.output_dir else args.output_dir
    )
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    shuffle = True

    # params for domain
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    heat_sink_origin = (-1, -0.3)
    nr_heat_sink_fins = 3
    gap = 0.15 + 0.1
    heat_sink_length = 1.0
    heat_sink_fin_thickness = 0.1
    inlet_vel = 1.5
    heat_sink_temp = 350
    base_temp = 293.498
    nu = 0.01
    diffusivity = 0.01 / 5
    max_distance = (channel_width[1] - channel_width[0]) / 2

    # define geometry
    channel = ppsci.geometry.Channel(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )

    heat_sink = ppsci.geometry.Rectangle(
        heat_sink_origin,
        (
            heat_sink_origin[0] + heat_sink_length,
            heat_sink_origin[1] + heat_sink_fin_thickness,
        ),
    )

    for i in range(1, nr_heat_sink_fins):
        heat_sink_origin = (heat_sink_origin[0], heat_sink_origin[1] + gap)
        fin = ppsci.geometry.Rectangle(
            heat_sink_origin,
            (
                heat_sink_origin[0] + heat_sink_length,
                heat_sink_origin[1] + heat_sink_fin_thickness,
            ),
        )
        heat_sink = heat_sink + fin

    geo = channel - heat_sink

    geo_inlet = ppsci.geometry.Line(
        (channel_length[0], channel_width[0]), (channel_length[0], channel_width[1]), (-1,0)
    )
    geo_outlet = ppsci.geometry.Line(
        (channel_length[1], channel_width[0]), (channel_length[1], channel_width[1]), (1,0)
    )

    integral_line = ppsci.geometry.Line(
        (0, channel_width[0]),
        (0, channel_width[1]),
        (1,0)
    )

    equation = ppsci.utils.misc.Prettydefaultdict()
    equation["ZeroEquation"] = ppsci.equation.ZeroEquation(0.01, max_distance)
    equation["NavierStokes"] = ppsci.equation.NavierStokes(equation["ZeroEquation"].expr, 1.0, 2, False, True)
    equation["AdvectionDiffusion"] = ppsci.equation.AdvectionDiffusion("c", diffusivity, 1.0, 0, 2, False)
    equation["GradNormal"] = ppsci.equation.GradNormal(grad_var="c", dim=2, time=False)
    equation["NormalDotVec"] = ppsci.equation.NormalDotVec(("u", "v"))

    ITERS_PER_EPOCH = 1000

    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": shuffle,
            "drop_last": True,
        },
        "num_workers": 1,
    }

    def parabola(input, inter_1=channel_width[0], inter_2=channel_width[1], height=inlet_vel):
        x = input["y"]
        factor = (4 * height) / (-(inter_1**2) - inter_2**2 + 2 * inter_1 * inter_2)
        return factor * (x - inter_1) * (x - inter_2)

    constraint_inlet = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "c": lambda d: d["c"]},
        {"u": parabola, "v": 0, "c": 0},
        geo_inlet,
        {**train_dataloader_cfg, "batch_size": 64},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={"u": 1, "v": 1, "c": 1},
        name="inlet"
    )
    
    constraint_outlet = ppsci.constraint.BoundaryConstraint(
        {"p": lambda d: d["p"]},
        {"p": 0},
        geo_outlet,
        {**train_dataloader_cfg, "batch_size": 64},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={"p": 1},
        name="outlet"
    )

    constraint_hs_wall = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "c": lambda d: d["c"]},
        {"u": 0, "v": 0, "c": (heat_sink_temp - base_temp) / 273.15},
        heat_sink,
        {**train_dataloader_cfg, "batch_size": 500},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={"u": 1, "v": 1, "c": 1},
        name="hs_wall"
    )

    constraint_channel_wall = ppsci.constraint.BoundaryConstraint(
        {
            "u": lambda d: d["u"],
            "v": lambda d: d["v"], 
            "normal_gradient_c": 
                lambda d: equation["GradNormal"].equations["normal_gradient_c"](d)},
        {"u": 0, "v": 0, "normal_gradient_c": 0},
        channel,
        {**train_dataloader_cfg, "batch_size": 2500},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={"u": 1, "v": 1, "normal_gradient_c": 1},
        name="channel_wall"
    )

    constraint_flow_interior = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        geo,
        {**train_dataloader_cfg, "batch_size": 4800},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={
            "continuity": "sdf",
            "momentum_x": "sdf",
            "momentum_y": "sdf"
        },
        name="interior_flow"
    )

    constraint_heat_interior = ppsci.constraint.InteriorConstraint(
        equation["AdvectionDiffusion"].equations,
        {"advection_diffusion_c": 0},
        geo,
        {**train_dataloader_cfg, "batch_size": 4800},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={
            "advection_diffusion_c": 1.0,
        },
        name="interior_heat"
    )

    # integral continuity
    class Integral_translate():
        def __init__(self):
            self.trans = 0
            self.trans_array = 5 * np.random.random((128 * 100,)) - 2.5
        
        def __call__(self, x, y):
            n = x.shape[0]
            x = x + self.trans
            if self.trans > heat_sink_origin[0] and self.trans < heat_sink_origin[0] + heat_sink_length:
                logic_sdf = geo.sdf_func(np.hstack((x.reshape(n,1), y.reshape(n,1)))) <= 0
            else:
                logic_sdf = np.full(n, True)
            return logic_sdf
        
        def set_trans(self, index):
            self.trans = self.trans_array[index]
            return self.trans
    
    integral_criteria = Integral_translate()
    integral_continuity = ppsci.constraint.IntegralConstraint(
        equation["NormalDotVec"].equations,
        {"normal_dot_vel": 1},
        integral_line,
        {**train_dataloader_cfg, "batch_size": 4, "integral_batch_size":128},
        ppsci.loss.IntegralLoss("sum"),
        criteria=integral_criteria,
        weight_dict={"normal_dot_vel": 0.1},
        name="integral_continuity",
    )

    constraint = {
        constraint_inlet.name: constraint_inlet,
        constraint_outlet.name: constraint_outlet,
        constraint_hs_wall.name: constraint_hs_wall,
        constraint_channel_wall.name: constraint_channel_wall,
        constraint_flow_interior.name: constraint_flow_interior,
        constraint_heat_interior.name: constraint_heat_interior,
        integral_continuity.name: integral_continuity
    }

    model_flow = ppsci.arch.MLP(
        ("x", "y"), ("u", "v", "p"), 6, 512, "silu", weight_norm=True)

    model_heat = ppsci.arch.MLP(
        ("x", "y"), ( "c"), 6, 512, "silu", weight_norm=True)

    model = ppsci.arch.ModelList((model_flow, model_heat))

    # set training hyper-parameters
    EPOCHS = 500 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        0.95,
        5000,
        by_epoch=False,
    )()

    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # add validation data
    mapping = {
        "x" : "Points:0",
        "y" : "Points:1",
        "u" : "U:0",
        "v" : "U:1",
        "p": "p",
        "sdf": "d",
        "nu": "nuT",
        "c": "T",
    }
    openfoam_var = ppsci.utils.reader.load_csv_file(
        "./data/openfoam/heat_sink_zeroEq_Pr5_mesh20.csv",
        mapping.keys(),
        mapping
    )
    openfoam_var["nu"] += nu
    openfoam_var["c"] += -base_temp
    openfoam_var["c"] /= 273.15
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value
        for key, value in openfoam_var.items()
        if key in ["u", "v", "p", "c"]  # Removing "nu"
    }

    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": openfoam_invar_numpy,
            "label": openfoam_outvar_numpy,
            "weight": {k: np.ones_like(v) for k, v in openfoam_outvar_numpy.items()},
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": shuffle,
            "drop_last": True,
        },
    }

    openfoam_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": 128},
        None,
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "p": lambda out: out["p"],
            "c": lambda out: out["c"],
        },
        metric={"MSE": ppsci.metric.L2Rel()},
        name="openfoam_validator",
    )

    F_MTR = "./data/monitor"
    input_global, _ = ppsci.utils.load_vtk_file(
        osp.join(F_MTR, "monitor_global.vtu"), 
        input_keys=('x','y'), 
        input_keys_patch=("sdf","area")
        )
    input_global['sdf__x'] = np.zeros_like(input_global['sdf'])
    input_global['sdf__y'] = np.zeros_like(input_global['sdf'])
    input_force, _ = ppsci.utils.load_vtk_file(
        osp.join(F_MTR, "monitor_force.vtu"), 
        input_keys=('x','y'), 
        input_keys_patch=("area", "normal_x", "normal_y")
        )
    input_peakT, _ = ppsci.utils.load_vtk_file(
        osp.join(F_MTR, "monitor_peakT.vtu"), 
        input_keys=('x','y'), 
        input_keys_patch=("area", "normal_x", "normal_y")
        )

    eval_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "label":{"dummy" : np.zeros(100, paddle.get_default_dtype())},
            "weight":{"dummy" : np.zeros(100, paddle.get_default_dtype())}
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": shuffle,
            "drop_last": True,
        },
        "batch_size": 100
    }

    eval_cfg["dataset"]["input"] = input_global
    eq_validator = ppsci.validate.SupervisedValidator(
        eval_cfg,
        None,
        {
            "continuity": equation["NavierStokes"].equations["continuity"],
            "momentum_x": equation["NavierStokes"].equations["momentum_x"],
            "momentum_y": equation["NavierStokes"].equations["momentum_y"]
        },
        {
            "global_monitor":lambda var, _: 
            {
                "mass_imbalance":
                paddle.sum(var["area"] * paddle.abs(var["continuity"])),
                "momentum_imbalance": 
                paddle.sum(var["area"] * (paddle.abs(var["momentum_x"]) + paddle.abs(var["momentum_y"])))
            }
        },
        "eq_validator",
    )

    eval_cfg["dataset"]["input"] = input_force
    force_validator = ppsci.validate.SupervisedValidator(
        eval_cfg,
        None,
        {"u": lambda out: out["u"]},
        {
            "heat_sink_force":lambda var, _: {
                "force_x": paddle.sum(var["normal_x"] * var["area"] * var["p"]),
                "force_y": paddle.sum(var["normal_y"] * var["area"] * var["p"])
            }
        },
        name="force_validator",
    )

    eval_cfg["dataset"]["input"] = input_peakT
    temperature_validator = ppsci.validate.SupervisedValidator(
        eval_cfg,
        None,
        {"u": lambda out: out["u"]},
        {"peak_T": lambda var, _:{"peak_T":paddle.max(var["c"])}},
        "temperature_validator",
    )

    validator = {
        openfoam_validator.name: openfoam_validator,
        eq_validator.name: eq_validator,
        force_validator.name: force_validator,
        temperature_validator.name: temperature_validator
    }

    visualizer = {"visualizer":ppsci.visualize.VisualizerVtu(
        openfoam_invar_numpy,
        {"pred_u": lambda d: d["u"], "pred_v": lambda d: d["v"], "pred_p": lambda d: d["p"], "pred_c": lambda d: d["c"]},
        batch_size=128,
        prefix="PaddleScience")}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=1,
        eval_during_train=True,
        log_freq=1,
        eval_freq=1,
        equation=equation,
        validator=validator,
        visualizer=visualizer
    )

    solver.train()
    solver.visualize()

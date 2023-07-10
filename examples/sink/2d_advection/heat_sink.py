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
Reference: https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/advanced/conjugate_heat_transfer.html#introduction
"""

import os
import ppsci
import paddle
import numpy as np
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    OUTPUT_DIR = (
        "./output_sink_0708" if not args.output_dir else args.output_dir
    )
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    os.chdir("/workspace/wangguan/PaddleScience_2D_Sink/examples/sink/2d_advection")
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

    # define geometry
    channel = ppsci.geometry.Rectangle(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[1])
    )

    channel_wall_top = ppsci.geometry.Line(
        (channel_length[0], channel_width[1]), (channel_length[1], channel_width[1]), (0,1)
    )

    channel_wall_bottom = ppsci.geometry.Line(
        (channel_length[0], channel_width[0]), (channel_length[1], channel_width[0]), (0,1)
    )

    channel_wall = channel_wall_top + channel_wall_bottom


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

    zero_eq =  ppsci.equation.ZeroEquation(0.01, (channel_width[1] - channel_width[0]) / 2)

    equation = {
        "ZeroEquation": zero_eq,
        "NavierStokes": ppsci.equation.NavierStokes(zero_eq.expr, 1.0, 2, False),
        "AdvectionDiffusion": ppsci.equation.AdvectionDiffusion(temperature="c", diffusivity=diffusivity,rho=1.0, dim=2, time=False),
        "GradNormal": ppsci.equation.GradNormal("c", dim=2, time=False),
        "NormalDotVec": ppsci.equation.NormalDotVec(("u", "v"))
        }

    ITERS_PER_EPOCH = 1000

    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
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
        name="inlet",
        out_dir = "./data/constraints/bc_inlet.vtu"
    )
    
    constraint_outlet = ppsci.constraint.BoundaryConstraint(
        {"p": lambda d: d["p"]},
        {"p": 0},
        geo_outlet,
        {**train_dataloader_cfg, "batch_size": 64},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={"p": 1},
        name="outlet",
        out_dir = "./data/constraints/bc_outlet.vtu"
    )

    constraint_hs_wall = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "c": lambda d: d["c"]},
        {"u": 0, "v": 0, "c": (heat_sink_temp - base_temp) / 273.15},
        heat_sink,
        {**train_dataloader_cfg, "batch_size": 500},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        weight_dict={"u": 1, "v": 1, "c": 1},
        name="hs_wall",
        out_dir = "./data/constraints/bc_hs_wall.vtu"
    )

    constraint_channel_wall = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "normal_gradient_c": lambda d: equation["GradNormal"].equations["normal_gradient_c"](d)},
        {"u": 0, "v": 0, "normal_gradient_c": 0},
        channel_wall, #channel,
        {**train_dataloader_cfg, "batch_size": 2500},
        ppsci.loss.MSELoss("sum"),
        criteria=None,
        # criteria=lambda x, y: (
        #     ( y == channel_width[0])
        #     | (y == channel_width[1])
        # ),
        weight_dict={"u": 1, "v": 1, "normal_gradient_c": 1},
        name="channel_wall",
        out_dir = "./data/constraints/bc_channel_wall.vtu"
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
        name="interior_flow",
        out_dir = "./data/inter_interior_flow.vtu",
        input_keys_patch=('sdf', 'sdf__x','sdf__y', 'area')
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
        name="interior_heat",
        out_dir = "./data/inter_interior_heat.vtu",
        input_keys_patch=('sdf', 'area')
    )
    # integral continuity
    class Integral_translate():
        def __init__(self):
            self.trans = 0
            self.trans_array = 5 * np.random.random((128 * 100,)) - 2.5
            # self.trans_array = np.linspace(-1.5,1.5, 15)
        
        def __call__(self, x, y):
            n = x.shape[0]
            x = x + self.trans
            if self.trans > -1 and self.trans < 0:
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

    folder = r"/workspace/wangguan/PaddleScience_2D_Sink/examples/sink/2d_advection/data/modulus/net_params/"

    net_params_f = paddle.load(folder + "flow_net_params.pdparams")
    net_params_t = paddle.load(folder + "thermal_net_params.pdparams")

    model_flow = ppsci.arch.MLP(
        ("x", "y"), ("u", "v", "p"), 6, 512, "silu", weight_norm=True, weight_init_v=net_params_f['w'], weight_init_g=net_params_f['w_grad'], bias_init=net_params_f['b'])

    model_heat = ppsci.arch.MLP(
        ("x", "y"), ( "c"), 6, 512, "silu", weight_norm=True, weight_init_v=net_params_t['w'], weight_init_g=net_params_t['w_grad'], bias_init=net_params_t['b'])

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
            "shuffle": True,
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
    # ppsci.visualize.save_vtu_to_mesh(out_dir, input, ("x","y"), ("sdf",))
    input_global, _ = ppsci.utils.load_vtk_file("./data/modulus/constraints/monitor_global.vtu", input_keys=('x','y'), input_keys_patch=("sdf","area"))
    input_global['sdf__x'] = np.zeros_like(input_global['sdf'])
    input_global['sdf__y'] = np.zeros_like(input_global['sdf'])

    input_force, _ = ppsci.utils.load_vtk_file("./data/modulus/constraints/monitor_force.vtu", input_keys=('x','y'), input_keys_patch=("area", "normal_x", "normal_y"))
    input_peakT, _ = ppsci.utils.load_vtk_file("./data/modulus/constraints/monitor_peakT.vtu", input_keys=('x','y'), input_keys_patch=("area", "normal_x", "normal_y"))
    
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    vtp_file_path = "./data/modulus/constraints/validator_modulus.vtp"
    vtk_reader = vtk.vtkXMLPolyDataReader()
    vtk_reader.SetFileName(vtp_file_path)
    vtk_reader.Update()
    vtk_poly_data = vtk_reader.GetOutput()
    vtk_coord = vtk_poly_data.GetPoints()
    vtk_array = vtk_coord.GetData()
    point_coordinates = vtk_to_numpy(vtk_array)
    vtk_points_data = vtk_poly_data.GetPointData()
    n_vtp = point_coordinates.shape[0]
    label_modulus = {
        'u':vtk_to_numpy(vtk_points_data.GetArray('pred_u')).reshape(n_vtp, 1),
        'v':vtk_to_numpy(vtk_points_data.GetArray('pred_v')).reshape(n_vtp, 1),
        'p':vtk_to_numpy(vtk_points_data.GetArray('pred_p')).reshape(n_vtp, 1),
        'c':vtk_to_numpy(vtk_points_data.GetArray('pred_c')).reshape(n_vtp, 1),
        }

    # input_modulus, label_modulus = ppsci.utils.load_vtk_file("./data/modulus/constraints/validator_modulus.vtp", input_keys=('x','y'), input_keys_patch=(), label_keys=("u", "v", "p", "c"))



    eval_dataloader_cfg_global= {
        "dataset": {
            "name": "InputDataset",
            "input": input_global #geo.sample_interior(100),
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": True,
        },
    }

    eval_dataloader_cfg_force = {
        "dataset": {
            "name": "InputDataset",
            "input": input_force#heat_sink.sample_boundary(100),
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": True,
        },
    }

    eval_dataloader_cfg_peakT = {
        "dataset": {
            "name": "InputDataset",
            "input": input_peakT#heat_sink.sample_boundary(100),
        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": True,
        },
    }

    eval_dataloader_cfg_modulus= {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": openfoam_invar_numpy, #geo.sample_interior(100),
            "label": label_modulus,
            "weight": {k: np.ones_like(v) for k, v in label_modulus.items()},

        },
        "sampler": {
            "name": "BatchSampler",
            "shuffle": True,
            "drop_last": True,
        },
    }

    eq_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg_global, "batch_size": 100},
        output_expr={
            "continuity":equation["NavierStokes"].equations["continuity"],
            "momentum_x": equation["NavierStokes"].equations["momentum_x"],
            "momentum_y": equation["NavierStokes"].equations["momentum_y"]
        },
        metric={
            "global_monitor":lambda var, _: 
            {
                "mass_imbalance":
                paddle.sum(var["area"] * paddle.abs(var["continuity"])),
                "momentum_imbalance": 
                paddle.sum(var["area"] * (paddle.abs(var["momentum_x"]) + paddle.abs(var["momentum_y"])))
            }
        },
        name="eq_validator",
    )

    force_validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg={**eval_dataloader_cfg_force, "batch_size": 100},
        output_expr={"u": lambda out: out["u"]},
        metric={
            "heat_sink_force":lambda var, _: {
                "force_x": paddle.sum(var["normal_x"] * var["area"] * var["p"]),
                "force_y": paddle.sum(var["normal_y"] * var["area"] * var["p"])
            }
        },
        name="force_validator",
    )

    temperature_validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg={**eval_dataloader_cfg_peakT, "batch_size": 100},
        output_expr={"u": lambda out: out["u"]},
        metric={"peak_T": lambda var, _:{"peak_T":paddle.max(var["c"])}},
        name="temperature_validator",
    )

    modulus_validator = ppsci.validate.SupervisedValidator(
        dataloader_cfg={**eval_dataloader_cfg_modulus, "batch_size": 100},
        output_expr={"u": lambda out: out["u"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="modulus_validator",
    )

    validator = {
        eq_validator.name: eq_validator,
        force_validator.name: force_validator,
        temperature_validator.name: temperature_validator,
        modulus_validator.name: modulus_validator
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
        visualizer=visualizer,
        checkpoint_path="/workspace/wangguan/PaddleScience_2D_Sink/examples/sink/2d_advection/output_sink_0708/checkpoints/latest"
    )
    solver.visualize()
    solver.model.training = False
    solver.eval()
    # solver.train()





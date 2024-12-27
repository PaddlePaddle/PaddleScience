# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from os import path as osp

import equation as eq_func
import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from paddle.nn import initializer

import ppsci
from ppsci.metric import L2Rel
from ppsci.utils import logger
from ppsci.utils import reader


def train(cfg: DictConfig):
    # set equation
    E = paddle.create_parameter(
        shape=[],
        dtype=paddle.get_default_dtype(),
        default_initializer=initializer.Constant(0.0),
    )
    equation = {"Hooke": eq_func.Hooke(E=E, nu=cfg.nu, P=cfg.P, dim=3)}

    # set models
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,) + tuple(equation.values()))

    # set geometry
    heart = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    base = ppsci.geometry.Mesh(cfg.BASE_PATH)
    endo = ppsci.geometry.Mesh(cfg.ENDO_PATH)
    epi = ppsci.geometry.Mesh(cfg.EPI_PATH)
    geom = {"geo": heart, "base": base, "endo": endo, "epi": epi}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = heart.bounds

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
    bc_base = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["base"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_base},
        ppsci.loss.MSELoss("sum"),
        weight_dict=cfg.TRAIN.weight.bc_base,
        name="BC_BASE",
    )
    bc_endo = ppsci.constraint.BoundaryConstraint(
        equation["Hooke"].equations,
        {"traction_x": -cfg.P, "traction_y": -cfg.P, "traction_z": -cfg.P},
        geom["endo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_endo},
        ppsci.loss.MSELoss("sum"),
        weight_dict=cfg.TRAIN.weight.bc_endo,
        name="BC_ENDO",
    )
    bc_epi = ppsci.constraint.BoundaryConstraint(
        equation["Hooke"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["epi"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.bc_epi},
        ppsci.loss.MSELoss("sum"),
        weight_dict=cfg.TRAIN.weight.bc_endo,
        name="BC_EPI",
    )
    interior = ppsci.constraint.InteriorConstraint(
        equation["Hooke"].equations,
        {"hooke_x": 0, "hooke_y": 0, "hooke_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": cfg.TRAIN.batch_size.interior},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        weight_dict=cfg.TRAIN.weight.interior,
        name="INTERIOR",
    )
    data = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableCSVDataset",
                "file_path": cfg.DATA_PATH,
                "input_keys": ("x", "y", "z"),
                "label_keys": ("u", "v", "w"),
            },
        },
        ppsci.loss.MSELoss("sum"),
        name="DATA",
    )

    # wrap constraints together
    constraint = {
        bc_base.name: bc_base,
        bc_endo.name: bc_endo,
        bc_epi.name: bc_epi,
        interior.name: interior,
        data.name: data,
    }

    # set validator
    eval_data_dict = reader.load_csv_file(
        cfg.DATA_PATH,
        ("x", "y", "z", "u", "v", "w"),
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "u": "u",
            "v": "v",
            "w": "w",
        },
    )
    input_dict = {
        "x": eval_data_dict["x"],
        "y": eval_data_dict["y"],
        "z": eval_data_dict["z"],
    }
    label_dict = {
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
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size},
        ppsci.loss.MSELoss("mean"),
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="ref_u_v_w",
    )

    fake_input = np.full((1, 1), 1, dtype=np.float32)
    E_label = np.full((1, 1), cfg.E, dtype=np.float32)
    param_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "x": fake_input,
                    "y": fake_input,
                    "z": fake_input,
                },
                "label": {"E": E_label},
            },
            "batch_size": 1,
            "num_workers": 1,
        },
        ppsci.loss.MSELoss("mean"),
        {
            "E": lambda out: E.reshape([1, 1]),
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="param_E",
    )

    validator = {
        sup_validator.name: sup_validator,
        param_validator.name: param_validator,
    }

    # set visualizer(optional)
    visualizer = {
        "visualize_u_v_w": ppsci.visualize.VisualizerVtu(
            input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            batch_size=cfg.EVAL.batch_size,
            prefix="result_u_v_w",
        ),
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

    # train
    solver.train()
    # eval
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()
    # plot loss
    solver.plot_loss_history(by_epoch=True)

    # save parameter E separately
    paddle.save({"E": E}, osp.join(cfg.output_dir, "param_E.pdparams"))


def evaluate(cfg: DictConfig):
    # set models
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry
    heart = ppsci.geometry.Mesh(cfg.GEOM_PATH)
    base = ppsci.geometry.Mesh(cfg.BASE_PATH)
    endo = ppsci.geometry.Mesh(cfg.ENDO_PATH)
    epi = ppsci.geometry.Mesh(cfg.EPI_PATH)
    # test = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
    geom = {"geo": heart, "base": base, "endo": endo, "epi": epi}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = heart.bounds

    # set validator
    eval_data_dict = reader.load_csv_file(
        cfg.DATA_PATH,
        ("x", "y", "z", "u", "v", "w"),
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "u": "u",
            "v": "v",
            "w": "w",
        },
    )
    input_dict = {
        "x": eval_data_dict["x"],
        "y": eval_data_dict["y"],
        "z": eval_data_dict["z"],
    }
    label_dict = {
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
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": cfg.EVAL.batch_size},
        ppsci.loss.MSELoss("mean"),
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="ref_u_v_w",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    # add inferencer data endo
    samples_endo = geom["endo"].sample_boundary(
        cfg.EVAL.num_vis,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict_endo = {
        "x": samples_endo["x"],
        "y": samples_endo["y"],
        "z": samples_endo["z"],
    }
    visualizer_endo = ppsci.visualize.VisualizerVtu(
        pred_input_dict_endo,
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        prefix="vtu_u_v_w_endo",
    )
    # add inferencer data epi
    samples_epi = geom["epi"].sample_boundary(
        cfg.EVAL.num_vis,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict_epi = {
        "x": samples_epi["x"],
        "y": samples_epi["y"],
        "z": samples_epi["z"],
    }
    visualizer_epi = ppsci.visualize.VisualizerVtu(
        pred_input_dict_epi,
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        prefix="vtu_u_v_w_epi",
    )
    # add inferencer data
    samples_geom = geom["geo"].sample_interior(
        cfg.EVAL.num_vis,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )
    pred_input_dict_geom = {
        "x": samples_geom["x"],
        "y": samples_geom["y"],
        "z": samples_geom["z"],
    }
    visualizer_geom = ppsci.visualize.VisualizerVtu(
        pred_input_dict_geom,
        {
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "w": lambda out: out["w"],
        },
        prefix="vtu_u_v_w_geom",
    )

    # wrap visualizers together
    visualizer = {
        "vis_eval_endo": visualizer_endo,
        "visualizer_epi": visualizer_epi,
        "vis_eval_geom": visualizer_geom,
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        validator=validator,
        visualizer=visualizer,
        cfg=cfg,
    )
    # evaluate
    solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # evaluate E
    E_truth = paddle.to_tensor(cfg.E, dtype=paddle.get_default_dtype()).reshape([1, 1])
    E_pred = paddle.load(cfg.EVAL.param_E_path)["E"].reshape([1, 1])
    l2_error = L2Rel()({"E": E_pred}, {"E": E_truth})["E"]
    logger.info(
        f"E_truth: {cfg.E}, E_pred: {float(E_pred)}, L2_Error: {float(l2_error)}"
    )


@hydra.main(version_base=None, config_path="./conf", config_name="inverse.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

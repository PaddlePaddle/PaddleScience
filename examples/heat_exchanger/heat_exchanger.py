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


from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # Set dimensionless calculation parameters
    DL = cfg.DL  # lenth of the domain
    cp_c = cfg.cp_c  # specific heat capacity of cold boundary
    cp_h = cfg.cp_h  # specific heat capacity of hot boundary
    cp_w = cfg.cp_w  # specific heat capacity of wall
    v_h = cfg.v_h  # flow rate of hot boundary
    v_c = cfg.v_c  # flow rate of cold boundary
    alpha_h = (
        cfg.alpha_h
    )  # surface efficiency*heat transfer coefficient*heat transfer area of hot boundary
    alpha_c = (
        cfg.alpha_c
    )  # surface efficiency*heat transfer coefficient*heat transfer area of cold boundary
    L = cfg.L  # flow length
    M = cfg.M  # heat transfer structural quality
    T_hin = cfg.T_hin  # initial temperature of hot boundary
    T_cin = cfg.T_cin  # initial temperature of cold boundary
    T_win = cfg.T_win  # initial temperature of wall
    w_h = alpha_h / (M * cp_w)
    w_c = alpha_c / (M * cp_w)

    # set model
    model = ppsci.arch.DeepONets(**cfg.MODEL)

    # pde/bc constraint use t1~tn, initial constraint use t0
    NPOINT, NTIME = cfg.NPOINT, cfg.NTIME
    NQM = cfg.NQM  # Number of branch network samples

    # set time-geometry
    timestamps = np.linspace(0.0, 2, NTIME + 1, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(0.0, 1, timestamps=timestamps),
            ppsci.geometry.Interval(0, DL),
        )
    }

    # Generate train data and eval data
    visu_mat = geom["time_rect"].sample_interior(NPOINT * NTIME, evenly=True)
    data_h = np.random.rand(NQM).reshape([-1, 1]) * 2
    data_c = np.random.rand(NQM).reshape([-1, 1]) * 2
    data_h = data_h.astype("float32")
    data_c = data_c.astype("float32")
    test_h = np.random.rand(1).reshape([-1, 1]).astype("float32")
    test_c = np.random.rand(1).reshape([-1, 1]).astype("float32")
    # rearrange train data and eval data
    points = visu_mat.copy()
    points["t"] = np.repeat(points["t"], NQM, axis=0)
    points["x"] = np.repeat(points["x"], NQM, axis=0)
    points["qm_h"] = np.tile(data_h, (NPOINT * NTIME, 1))
    points["t"] = np.repeat(points["t"], NQM, axis=0)
    points["x"] = np.repeat(points["x"], NQM, axis=0)
    points["qm_h"] = np.repeat(points["qm_h"], NQM, axis=0)
    points["qm_c"] = np.tile(data_c, (NPOINT * NTIME * NQM, 1))
    visu_mat["qm_h"] = np.tile(test_h, (NPOINT * NTIME, 1))
    visu_mat["qm_c"] = np.tile(test_c, (NPOINT * NTIME, 1))

    left_indices = visu_mat["x"] == 0
    right_indices = visu_mat["x"] == DL
    interior_indices = (visu_mat["x"] != 0) & (visu_mat["x"] != DL)
    left_indices = np.where(left_indices)
    right_indices = np.where(right_indices)
    interior_indices = np.where(interior_indices)

    left_indices1 = points["x"] == 0
    right_indices1 = points["x"] == DL
    interior_indices1 = (points["x"] != 0) & (points["x"] != DL)
    initial_indices1 = points["t"] == points["t"][0]
    left_indices1 = np.where(left_indices1)
    right_indices1 = np.where(right_indices1)
    interior_indices1 = np.where(interior_indices1)
    initial_indices1 = np.where(initial_indices1)

    # Classification train data
    left_data = {
        "x": points["x"][left_indices1[0]],
        "t": points["t"][left_indices1[0]],
        "qm_h": points["qm_h"][left_indices1[0]],
        "qm_c": points["qm_c"][left_indices1[0]],
    }
    right_data = {
        "x": points["x"][right_indices1[0]],
        "t": points["t"][right_indices1[0]],
        "qm_h": points["qm_h"][right_indices1[0]],
        "qm_c": points["qm_c"][right_indices1[0]],
    }
    interior_data = {
        "x": points["x"],
        "t": points["t"],
        "qm_h": points["qm_h"],
        "qm_c": points["qm_c"],
    }
    initial_data = {
        "x": points["x"][initial_indices1[0]],
        "t": points["t"][initial_indices1[0]] * 0,
        "qm_h": points["qm_h"][initial_indices1[0]],
        "qm_c": points["qm_c"][initial_indices1[0]],
    }
    # Classification eval data
    test_left_data = {
        "x": visu_mat["x"][left_indices[0]],
        "t": visu_mat["t"][left_indices[0]],
        "qm_h": visu_mat["qm_h"][left_indices[0]],
        "qm_c": visu_mat["qm_c"][left_indices[0]],
    }
    test_right_data = {
        "x": visu_mat["x"][right_indices[0]],
        "t": visu_mat["t"][right_indices[0]],
        "qm_h": visu_mat["qm_h"][right_indices[0]],
        "qm_c": visu_mat["qm_c"][right_indices[0]],
    }
    test_interior_data = {
        "x": visu_mat["x"],
        "t": visu_mat["t"],
        "qm_h": visu_mat["qm_h"],
        "qm_c": visu_mat["qm_c"],
    }

    # set equation
    equation = {
        "heat_exchanger": ppsci.equation.HeatExchanger(
            alpha_h / (L * cp_h), alpha_c / (L * cp_c), v_h, v_c, w_h, w_c
        )
    }

    # set constraint
    BC_label = {
        "T_h": np.zeros([left_data["x"].shape[0], 1], dtype="float32"),
    }
    interior_label = {
        "heat_boundary": np.zeros([interior_data["x"].shape[0], 1], dtype="float32"),
        "cold_boundary": np.zeros([interior_data["x"].shape[0], 1], dtype="float32"),
        "wall": np.zeros([interior_data["x"].shape[0], 1], dtype="float32"),
    }
    initial_label = {
        "T_h": np.zeros([initial_data["x"].shape[0], 1], dtype="float32"),
        "T_c": np.zeros([initial_data["x"].shape[0], 1], dtype="float32"),
        "T_w": np.zeros([initial_data["x"].shape[0], 1], dtype="float32"),
    }

    left_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": left_data,
                "label": BC_label,
                "weight": {
                    "T_h": cfg.TRAIN.weight
                    * np.ones([left_data["x"].shape[0], 1], dtype="float32")
                },
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_h"] - T_hin,
        },
        name="left_sup",
    )
    right_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": right_data,
                "label": BC_label,
                "weight": {
                    "T_h": cfg.TRAIN.weight
                    * np.ones([right_data["x"].shape[0], 1], dtype="float32")
                },
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_c"] - T_cin,
        },
        name="right_sup",
    )
    interior_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": interior_data,
                "label": interior_label,
                "weight": {
                    "heat_boundary": 1
                    * np.ones([interior_data["x"].shape[0], 1], dtype="float32"),
                    "cold_boundary": 1
                    * np.ones([interior_data["x"].shape[0], 1], dtype="float32"),
                    "wall": cfg.TRAIN.weight
                    * np.ones([interior_data["x"].shape[0], 1], dtype="float32"),
                },
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr=equation["heat_exchanger"].equations,
        name="interior_sup",
    )
    initial_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": initial_data,
                "label": initial_label,
                "weight": {
                    "T_h": 1
                    * np.ones([initial_data["x"].shape[0], 1], dtype="float32"),
                    "T_c": 1
                    * np.ones([initial_data["x"].shape[0], 1], dtype="float32"),
                    "T_w": cfg.TRAIN.weight
                    * np.ones([initial_data["x"].shape[0], 1], dtype="float32"),
                },
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_h"] - T_hin,
            "T_c": lambda out: out["T_c"] - T_cin,
            "T_w": lambda out: out["T_w"] - T_win,
        },
        name="initial_sup",
    )
    # wrap constraints together
    constraint = {
        left_sup_constraint.name: left_sup_constraint,
        right_sup_constraint.name: right_sup_constraint,
        interior_sup_constraint.name: interior_sup_constraint,
        initial_sup_constraint.name: initial_sup_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)((model,))

    # set validator
    test_BC_label = {
        "T_h": np.zeros([test_left_data["x"].shape[0], 1], dtype="float32"),
    }
    test_interior_label = {
        "heat_boundary": np.zeros(
            [test_interior_data["x"].shape[0], 1], dtype="float32"
        ),
        "cold_boundary": np.zeros(
            [test_interior_data["x"].shape[0], 1], dtype="float32"
        ),
        "wall": np.zeros([test_interior_data["x"].shape[0], 1], dtype="float32"),
    }
    left_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_left_data,
                "label": test_BC_label,
            },
            "batch_size": NTIME,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_h"] - T_hin,
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="left_mse",
    )
    right_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_right_data,
                "label": test_BC_label,
            },
            "batch_size": NTIME,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_c"] - T_cin,
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="right_mse",
    )
    interior_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_interior_data,
                "label": test_interior_label,
            },
            "batch_size": NTIME,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr=equation["heat_exchanger"].equations,
        metric={"MSE": ppsci.metric.MSE()},
        name="interior_mse",
    )
    validator = {
        left_validator.name: left_validator,
        right_validator.name: right_validator,
        interior_validator.name: interior_validator,
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # plotting iteration/epoch-loss curve.
    solver.plot_loss_history()

    # visualize prediction after finished training
    visu_mat["qm_c"] = visu_mat["qm_c"] * 0 + cfg.qm_h
    visu_mat["qm_h"] = visu_mat["qm_c"] * 0 + cfg.qm_c
    pred = solver.predict(visu_mat)
    x = visu_mat["x"][:NPOINT]
    # plot temperature of heat boundary
    plt.figure()
    y = pred["T_h"][:101].numpy() * 0 + T_hin
    plt.plot(x, y, label="t = 0.0 s")
    for i in range(10):
        y = pred["T_h"][101 * i * 2 : 101 * (i * 2 + 1)].numpy()
        plt.plot(x, y, label=f"t = {(i+1)*0.1:,.1f} s")
    plt.xlabel("A")
    plt.ylabel(r"$T_h$")
    plt.legend()
    plt.grid()
    plt.savefig("T_h.png")
    # plot temperature of cold boundary
    plt.figure()
    y = pred["T_c"][:101].numpy() * 0 + T_cin
    plt.plot(x, y, label="t = 0.0 s")
    for i in range(10):
        y = pred["T_c"][101 * i * 2 : 101 * (i * 2 + 1)].numpy()
        plt.plot(x, y, label="t = {(i+1)*0.1:,.1f} s")
    plt.xlabel("A")
    plt.ylabel(r"$T_c$")
    plt.legend()
    plt.grid()
    plt.savefig("T_c.png")
    # plot temperature of wall
    plt.figure()
    y = pred["T_w"][:101].numpy() * 0 + T_win
    plt.plot(x, y, label="t = 0.0 s")
    for i in range(10):
        y = pred["T_w"][101 * i * 2 : 101 * (i * 2 + 1)].numpy()
        plt.plot(x, y, label="t = {(i+1)*0.1:,.1f} s")
    plt.xlabel("A")
    plt.ylabel(r"$T_w$")
    plt.legend()
    plt.grid()
    plt.savefig("T_w.png")
    # plot the heat exchanger efficiency as a function of time.
    plt.figure()
    qm_min = np.min((visu_mat["qm_h"][0], visu_mat["qm_c"][0]))
    eta = (
        visu_mat["qm_h"][0]
        * (pred["T_h"][::101] - pred["T_h"][100::101])
        / (qm_min * (pred["T_h"][::101] - pred["T_c"][100::101]))
    ).numpy()
    x = list(range(1, NTIME + 1))
    plt.plot(x, eta)
    plt.xlabel("time")
    plt.ylabel(r"$\eta$")
    plt.grid()
    plt.savefig("eta.png")


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # Set dimensionless calculation parameters
    DL = cfg.DL  # lenth of the domain
    cp_c = cfg.cp_c  # specific heat capacity of cold boundary
    cp_h = cfg.cp_h  # specific heat capacity of hot boundary
    cp_w = cfg.cp_w  # specific heat capacity of wall
    v_h = cfg.v_h  # flow rate of hot boundary
    v_c = cfg.v_c  # flow rate of cold boundary
    alpha_h = (
        cfg.alpha_h
    )  # surface efficiency*heat transfer coefficient*heat transfer area of hot boundary
    alpha_c = (
        cfg.alpha_c
    )  # surface efficiency*heat transfer coefficient*heat transfer area of cold boundary
    L = cfg.L  # flow length
    M = cfg.M  # heat transfer structural quality
    T_hin = cfg.T_hin  # initial temperature of hot boundary
    T_cin = cfg.T_cin  # initial temperature of cold boundary
    T_win = cfg.T_win  # initial temperature of wall
    w_h = alpha_h / (M * cp_w)
    w_c = alpha_c / (M * cp_w)

    # set model
    model = ppsci.arch.DeepONets(**cfg.MODEL)

    # pde/bc constraint use t1~tn, initial constraint use t0
    NPOINT, NTIME = cfg.NPOINT, cfg.NTIME

    # set time-geometry
    timestamps = np.linspace(0.0, 2, NTIME + 1, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(0.0, 1, timestamps=timestamps),
            ppsci.geometry.Interval(0, DL),
        )
    }

    # Generate eval data
    visu_mat = geom["time_rect"].sample_interior(NPOINT * NTIME, evenly=True)
    test_h = np.random.rand(1).reshape([-1, 1]).astype("float32")
    test_c = np.random.rand(1).reshape([-1, 1]).astype("float32")
    # rearrange train data and eval data
    visu_mat["qm_h"] = np.tile(test_h, (NPOINT * NTIME, 1))
    visu_mat["qm_c"] = np.tile(test_c, (NPOINT * NTIME, 1))

    left_indices = visu_mat["x"] == 0
    right_indices = visu_mat["x"] == DL
    interior_indices = (visu_mat["x"] != 0) & (visu_mat["x"] != DL)
    left_indices = np.where(left_indices)
    right_indices = np.where(right_indices)
    interior_indices = np.where(interior_indices)

    # Classification eval data
    test_left_data = {
        "x": visu_mat["x"][left_indices[0]],
        "t": visu_mat["t"][left_indices[0]],
        "qm_h": visu_mat["qm_h"][left_indices[0]],
        "qm_c": visu_mat["qm_c"][left_indices[0]],
    }
    test_right_data = {
        "x": visu_mat["x"][right_indices[0]],
        "t": visu_mat["t"][right_indices[0]],
        "qm_h": visu_mat["qm_h"][right_indices[0]],
        "qm_c": visu_mat["qm_c"][right_indices[0]],
    }
    test_interior_data = {
        "x": visu_mat["x"],
        "t": visu_mat["t"],
        "qm_h": visu_mat["qm_h"],
        "qm_c": visu_mat["qm_c"],
    }

    # set equation
    equation = {
        "heat_exchanger": ppsci.equation.HeatExchanger(
            alpha_h / (L * cp_h), alpha_c / (L * cp_c), v_h, v_c, w_h, w_c
        )
    }

    # set validator
    test_BC_label = {
        "T_h": np.zeros([test_left_data["x"].shape[0], 1], dtype="float32"),
    }
    test_interior_label = {
        "heat_boundary": np.zeros(
            [test_interior_data["x"].shape[0], 1], dtype="float32"
        ),
        "cold_boundary": np.zeros(
            [test_interior_data["x"].shape[0], 1], dtype="float32"
        ),
        "wall": np.zeros([test_interior_data["x"].shape[0], 1], dtype="float32"),
    }
    left_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_left_data,
                "label": test_BC_label,
            },
            "batch_size": NTIME,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_h"] - T_hin,
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="left_mse",
    )
    right_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_right_data,
                "label": test_BC_label,
            },
            "batch_size": NTIME,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "T_h": lambda out: out["T_c"] - T_cin,
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="right_mse",
    )
    interior_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_interior_data,
                "label": test_interior_label,
            },
            "batch_size": NTIME,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        output_expr=equation["heat_exchanger"].equations,
        metric={"MSE": ppsci.metric.MSE()},
        name="interior_mse",
    )
    validator = {
        left_validator.name: left_validator,
        right_validator.name: right_validator,
        interior_validator.name: interior_validator,
    }

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()

    # visualize prediction after finished training
    visu_mat["qm_c"] = visu_mat["qm_c"] * 0 + cfg.qm_h
    visu_mat["qm_h"] = visu_mat["qm_c"] * 0 + cfg.qm_c
    pred = solver.predict(visu_mat)
    x = visu_mat["x"][:NPOINT]
    # plot temperature of heat boundary
    plt.figure()
    y = pred["T_h"][:101].numpy() * 0 + T_hin
    plt.plot(x, y, label="t = 0.0 s")
    for i in range(10):
        y = pred["T_h"][101 * i * 2 : 101 * (i * 2 + 1)].numpy()
        plt.plot(x, y, label="t = {(i+1)*0.1:,.1f} s")
    plt.xlabel("A")
    plt.ylabel(r"$T_h$")
    plt.legend()
    plt.grid()
    plt.savefig("T_h.png")
    # plot temperature of cold boundary
    plt.figure()
    y = pred["T_c"][:101].numpy() * 0 + T_cin
    plt.plot(x, y, label="t = 0.0 s")
    for i in range(10):
        y = pred["T_c"][101 * i * 2 : 101 * (i * 2 + 1)].numpy()
        plt.plot(x, y, label="t = {(i+1)*0.1:,.1f} s")
    plt.xlabel("A")
    plt.ylabel(r"$T_c$")
    plt.legend()
    plt.grid()
    plt.savefig("T_c.png")
    # plot temperature of wall
    plt.figure()
    y = pred["T_w"][:101].numpy() * 0 + T_win
    plt.plot(x, y, label="t = 0.0 s")
    for i in range(10):
        y = pred["T_w"][101 * i * 2 : 101 * (i * 2 + 1)].numpy()
        plt.plot(x, y, label="t = {(i+1)*0.1:,.1f} s")
    plt.xlabel("A")
    plt.ylabel(r"$T_w$")
    plt.legend()
    plt.grid()
    plt.savefig("T_w.png")
    # plot the heat exchanger efficiency as a function of time.
    plt.figure()
    qm_min = np.min((visu_mat["qm_h"][0], visu_mat["qm_c"][0]))
    eta = (
        visu_mat["qm_h"][0]
        * (pred["T_h"][::101] - pred["T_h"][100::101])
        / (qm_min * (pred["T_h"][::101] - pred["T_c"][100::101]))
    ).numpy()
    x = list(range(1, NTIME + 1))
    plt.plot(x, eta)
    plt.xlabel("time")
    plt.ylabel(r"$\eta$")
    plt.grid()
    plt.savefig("eta.png")


@hydra.main(version_base=None, config_path="./conf", config_name="heat_exchanger.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

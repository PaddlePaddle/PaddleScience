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
import numpy as np
import paddle
import scipy.fftpack
import scipy.io
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger


def fftind(size):
    """
    Returns the momentum indices for the 2D Fast Fourier Transform (FFT).

    Args:
        size (int): Size of the 2D array.

    Returns:
        numpy.ndarray: Array of momentum indices for the 2D FFT.
    """
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_ind = scipy.fftpack.fftshift(k_ind)
    return k_ind


def GRF(alpha=3.0, size=128, flag_normalize=True):
    """
    Generates a Gaussian random field(GRF) with a power law amplitude spectrum.

    Args:
        alpha (float, optional): Power law exponent. Defaults to 3.0.
        size (int, optional): Size of the output field. Defaults to 128.
        flag_normalize (bool, optional): Flag indicating whether to normalize the field. Defaults to True.

    Returns:
        numpy.ndarray: Generated Gaussian random field.
    """
    # Defines momentum indices
    k_idx = fftind(size)
    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 4.0)
    amplitude[0, 0] = 0
    # Draws a complex gaussian random noise with normal
    # (circular) distribution
    noise = np.random.normal(size=(size, size)) + 1j * np.random.normal(
        size=(size, size)
    )
    # To real space
    gfield = np.fft.ifft2(noise * amplitude).real
    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield / np.std(gfield)
    return gfield.reshape([1, -1])


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.ChipDeepONets(**cfg.MODEL)
    # set geometry
    NPOINT = cfg.NL * cfg.NW
    geom = {"rect": ppsci.geometry.Rectangle((0, 0), (cfg.DL, cfg.DW))}
    points = geom["rect"].sample_interior(NPOINT, evenly=True)

    # generate training data and validation data
    data_u = np.ones([1, (cfg.NL - 2) * (cfg.NW - 2)])
    data_BC = np.ones([1, NPOINT])
    data_u = np.vstack((data_u, np.zeros([1, (cfg.NL - 2) * (cfg.NW - 2)])))
    data_BC = np.vstack((data_BC, np.zeros([1, NPOINT])))
    for i in range(cfg.NU - 2):
        data_u = np.vstack((data_u, GRF(alpha=cfg.GRF.alpha, size=cfg.NL - 2)))
    for i in range(cfg.NBC - 2):
        data_BC = np.vstack((data_BC, GRF(alpha=cfg.GRF.alpha, size=cfg.NL)))
    data_u = data_u.astype("float32")
    data_BC = data_BC.astype("float32")
    test_u = GRF(alpha=4, size=cfg.NL).astype("float32")[0]

    boundary_indices = np.where(
        (
            (points["x"] == 0)
            | (points["x"] == cfg.DW)
            | (points["y"] == 0)
            | (points["y"] == cfg.DL)
        )
    )
    interior_indices = np.where(
        (
            (points["x"] != 0)
            & (points["x"] != cfg.DW)
            & (points["y"] != 0)
            & (points["y"] != cfg.DL)
        )
    )

    points["u"] = np.tile(test_u[interior_indices[0]], (NPOINT, 1))
    points["u_one"] = test_u.T.reshape([-1, 1])
    points["bc_data"] = np.tile(test_u[boundary_indices[0]], (NPOINT, 1))
    points["bc"] = np.zeros((NPOINT, 1), dtype="float32")

    top_indices = np.where(points["x"] == cfg.DW)
    down_indices = np.where(points["x"] == 0)
    left_indices = np.where(
        (points["y"] == 0) & (points["x"] != 0) & (points["x"] != cfg.DW)
    )
    right_indices = np.where(
        ((points["y"] == cfg.DL) & (points["x"] != 0) & (points["x"] != cfg.DW))
    )

    # generate validation data
    (
        test_top_data,
        test_down_data,
        test_left_data,
        test_right_data,
        test_interior_data,
    ) = [
        {
            "x": points["x"][indices_[0]],
            "y": points["y"][indices_[0]],
            "u": points["u"][indices_[0]],
            "u_one": points["u_one"][indices_[0]],
            "bc": points["bc"][indices_[0]],
            "bc_data": points["bc_data"][indices_[0]],
        }
        for indices_ in (
            top_indices,
            down_indices,
            left_indices,
            right_indices,
            interior_indices,
        )
    ]
    # generate train data
    top_data = {
        "x": test_top_data["x"],
        "y": test_top_data["y"],
        "u": data_u,
        "u_one": data_BC[:, top_indices[0]].T.reshape([-1, 1]),
        "bc": np.array([[0], [1], [2], [3]], dtype="float32"),
        "bc_data": data_BC[:, boundary_indices[0]],
    }
    down_data = {
        "x": test_down_data["x"],
        "y": test_down_data["y"],
        "u": data_u,
        "u_one": data_BC[:, down_indices[0]].T.reshape([-1, 1]),
        "bc": np.array([[0], [1], [2], [3]], dtype="float32"),
        "bc_data": data_BC[:, boundary_indices[0]],
    }
    left_data = {
        "x": test_left_data["x"],
        "y": test_left_data["y"],
        "u": data_u,
        "u_one": data_BC[:, left_indices[0]].T.reshape([-1, 1]),
        "bc": np.array([[0], [1], [2], [3]], dtype="float32"),
        "bc_data": data_BC[:, boundary_indices[0]],
    }
    right_data = {
        "x": test_right_data["x"],
        "y": test_right_data["y"],
        "u": data_u,
        "u_one": data_BC[:, right_indices[0]].T.reshape([-1, 1]),
        "bc": np.array([[0], [1], [2], [3]], dtype="float32"),
        "bc_data": data_BC[:, boundary_indices[0]],
    }
    interior_data = {
        "x": test_interior_data["x"],
        "y": test_interior_data["y"],
        "u": data_u,
        "u_one": data_u.T.reshape([-1, 1]),
        "bc": np.array([[0], [1], [2], [3]], dtype="float32"),
        "bc_data": data_BC[:, boundary_indices[0]],
    }

    # set constraint
    index = ("x", "u", "bc", "bc_data")
    label = {"chip": np.array([0], dtype="float32")}
    weight = {"chip": np.array([cfg.TRAIN.weight], dtype="float32")}
    top_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ChipHeatDataset",
                "input": top_data,
                "label": label,
                "index": index,
                "data_type": "bc_data",
                "weight": weight,
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
            "chip": lambda out: paddle.where(
                out["bc"] == 1,
                jacobian(out["T"], out["x"]) - out["u_one"],
                paddle.where(
                    out["bc"] == 0,
                    out["T"] - out["u_one"],
                    paddle.where(
                        out["bc"] == 2,
                        jacobian(out["T"], out["x"]) + out["u_one"] * (out["T"] - 1),
                        jacobian(out["T"], out["x"])
                        + out["u_one"]
                        * (out["T"] ** 2 - 1)
                        * (out["T"] ** 2 + 1)
                        * 5.6
                        / 50000,
                    ),
                ),
            )
        },
        name="top_sup",
    )
    down_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ChipHeatDataset",
                "input": down_data,
                "label": label,
                "index": index,
                "data_type": "bc_data",
                "weight": weight,
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
            "chip": lambda out: paddle.where(
                out["bc"] == 1,
                jacobian(out["T"], out["x"]) - out["u_one"],
                paddle.where(
                    out["bc"] == 0,
                    out["T"] - out["u_one"],
                    paddle.where(
                        out["bc"] == 2,
                        jacobian(out["T"], out["x"]) + out["u_one"] * (out["T"] - 1),
                        jacobian(out["T"], out["x"])
                        + out["u_one"]
                        * (out["T"] ** 2 - 1)
                        * (out["T"] ** 2 + 1)
                        * 5.6
                        / 50000,
                    ),
                ),
            )
        },
        name="down_sup",
    )
    left_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ChipHeatDataset",
                "input": left_data,
                "label": label,
                "index": index,
                "data_type": "bc_data",
                "weight": weight,
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
            "chip": lambda out: paddle.where(
                out["bc"] == 1,
                jacobian(out["T"], out["y"]) - out["u_one"],
                paddle.where(
                    out["bc"] == 0,
                    out["T"] - out["u_one"],
                    paddle.where(
                        out["bc"] == 2,
                        jacobian(out["T"], out["y"]) + out["u_one"] * (out["T"] - 1),
                        jacobian(out["T"], out["y"])
                        + out["u_one"]
                        * (out["T"] ** 2 - 1)
                        * (out["T"] ** 2 + 1)
                        * 5.6
                        / 50000,
                    ),
                ),
            )
        },
        name="left_sup",
    )
    right_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ChipHeatDataset",
                "input": right_data,
                "label": label,
                "index": index,
                "data_type": "bc_data",
                "weight": weight,
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
            "chip": lambda out: paddle.where(
                out["bc"] == 1,
                jacobian(out["T"], out["y"]) - out["u_one"],
                paddle.where(
                    out["bc"] == 0,
                    out["T"] - out["u_one"],
                    paddle.where(
                        out["bc"] == 2,
                        jacobian(out["T"], out["y"]) + out["u_one"] * (out["T"] - 1),
                        jacobian(out["T"], out["y"])
                        + out["u_one"]
                        * (out["T"] ** 2 - 1)
                        * (out["T"] ** 2 + 1)
                        * 5.6
                        / 50000,
                    ),
                ),
            )
        },
        name="right_sup",
    )
    interior_sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ChipHeatDataset",
                "input": interior_data,
                "label": label,
                "index": index,
                "data_type": "u",
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
            "chip": lambda out: hessian(out["T"], out["x"])
            + hessian(out["T"], out["y"])
            + 100 * out["u_one"]
        },
        name="interior_sup",
    )
    # wrap constraints together
    constraint = {
        down_sup_constraint.name: down_sup_constraint,
        left_sup_constraint.name: left_sup_constraint,
        right_sup_constraint.name: right_sup_constraint,
        interior_sup_constraint.name: interior_sup_constraint,
        top_sup_constraint.name: top_sup_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    top_down_label = {"chip": np.zeros([cfg.NL, 1], dtype="float32")}
    left_right_label = {"chip": np.zeros([(cfg.NL - 2), 1], dtype="float32")}
    interior_label = {
        "thermal_condution": np.zeros(
            [test_interior_data["x"].shape[0], 1], dtype="float32"
        )
    }
    top_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_top_data,
                "label": top_down_label,
                "weight": {
                    "chip": np.full([cfg.NL, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": cfg.NL,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="top_mse",
    )
    down_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_down_data,
                "label": top_down_label,
                "weight": {
                    "chip": np.full([cfg.NL, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": cfg.NL,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="down_mse",
    )
    left_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_left_data,
                "label": left_right_label,
                "weight": {
                    "chip": np.full([cfg.NL - 2, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": (cfg.NL - 2),
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="left_mse",
    )
    right_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_right_data,
                "label": left_right_label,
                "weight": {
                    "chip": np.full([cfg.NL - 2, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": (cfg.NL - 2),
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="right_mse",
    )
    interior_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_interior_data,
                "label": interior_label,
            },
            "batch_size": cfg.TRAIN.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "thermal_condution": lambda out: (
                hessian(out["T"], out["x"]) + hessian(out["T"], out["y"])
            )
            + 100 * out["u_one"]
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="interior_mse",
    )
    validator = {
        down_validator.name: down_validator,
        left_validator.name: left_validator,
        right_validator.name: right_validator,
        top_validator.name: top_validator,
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
        validator=validator,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    pred_points = geom["rect"].sample_interior(NPOINT, evenly=True)
    pred_points["u"] = points["u"]
    pred_points["bc_data"] = np.zeros_like(points["bc_data"])
    pred_points["bc"] = np.repeat(
        np.array([[cfg.EVAL.bc_type]], dtype="float32"), NPOINT, axis=0
    )
    pred = solver.predict(pred_points)
    logger.message("Now saving visual result to: visual/result.vtu, please wait...")
    ppsci.visualize.save_vtu_from_dict(
        osp.join(cfg.output_dir, "visual/result.vtu"),
        {
            "x": pred_points["x"],
            "y": pred_points["y"],
            "T": pred["T"],
        },
        (
            "x",
            "y",
        ),
        ("T"),
    )


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.ChipDeepONets(**cfg.MODEL)
    # set geometry
    NPOINT = cfg.NL * cfg.NW
    geom = {"rect": ppsci.geometry.Rectangle((0, 0), (cfg.DL, cfg.DW))}
    points = geom["rect"].sample_interior(NPOINT, evenly=True)

    # generate validation data
    test_u = GRF(alpha=4, size=cfg.NL).astype("float32")[0]

    boundary_indices = np.where(
        (
            (points["x"] == 0)
            | (points["x"] == cfg.DW)
            | (points["y"] == 0)
            | (points["y"] == cfg.DL)
        )
    )
    interior_indices = np.where(
        (
            (points["x"] != 0)
            & (points["x"] != cfg.DW)
            & (points["y"] != 0)
            & (points["y"] != cfg.DL)
        )
    )

    points["u"] = np.tile(test_u[interior_indices[0]], (NPOINT, 1))
    points["u_one"] = test_u.T.reshape([-1, 1])
    points["bc_data"] = np.tile(test_u[boundary_indices[0]], (NPOINT, 1))
    points["bc"] = np.zeros((NPOINT, 1), dtype="float32")

    top_indices = np.where(points["x"] == cfg.DW)
    down_indices = np.where(points["x"] == 0)
    left_indices = np.where(
        (points["y"] == 0) & (points["x"] != 0) & (points["x"] != cfg.DW)
    )
    right_indices = np.where(
        ((points["y"] == cfg.DL) & (points["x"] != 0) & (points["x"] != cfg.DW))
    )

    # generate validation data
    (
        test_top_data,
        test_down_data,
        test_left_data,
        test_right_data,
        test_interior_data,
    ) = [
        {
            "x": points["x"][indices_[0]],
            "y": points["y"][indices_[0]],
            "u": points["u"][indices_[0]],
            "u_one": points["u_one"][indices_[0]],
            "bc": points["bc"][indices_[0]],
            "bc_data": points["bc_data"][indices_[0]],
        }
        for indices_ in (
            top_indices,
            down_indices,
            left_indices,
            right_indices,
            interior_indices,
        )
    ]

    # set validator
    top_down_label = {"chip": np.zeros([cfg.NL, 1], dtype="float32")}
    left_right_label = {"chip": np.zeros([(cfg.NL - 2), 1], dtype="float32")}
    interior_label = {
        "thermal_condution": np.zeros(
            [test_interior_data["x"].shape[0], 1], dtype="float32"
        )
    }
    top_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_top_data,
                "label": top_down_label,
                "weight": {
                    "chip": np.full([cfg.NL, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": cfg.NL,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="top_mse",
    )
    down_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_down_data,
                "label": top_down_label,
                "weight": {
                    "chip": np.full([cfg.NL, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": cfg.NL,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="down_mse",
    )
    left_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_left_data,
                "label": left_right_label,
                "weight": {
                    "chip": np.full([cfg.NL - 2, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": (cfg.NL - 2),
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="left_mse",
    )
    right_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_right_data,
                "label": left_right_label,
                "weight": {
                    "chip": np.full([cfg.NL - 2, 1], cfg.TRAIN.weight, dtype="float32")
                },
            },
            "batch_size": (cfg.NL - 2),
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={"chip": lambda out: out["T"] - out["u_one"]},
        metric={"MSE": ppsci.metric.MSE()},
        name="right_mse",
    )
    interior_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_interior_data,
                "label": interior_label,
            },
            "batch_size": cfg.TRAIN.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        output_expr={
            "thermal_condution": lambda out: (
                hessian(out["T"], out["x"]) + hessian(out["T"], out["y"])
            )
            + 100 * out["u_one"]
        },
        metric={"MSE": ppsci.metric.MSE()},
        name="interior_mse",
    )
    validator = {
        down_validator.name: down_validator,
        left_validator.name: left_validator,
        right_validator.name: right_validator,
        top_validator.name: top_validator,
        interior_validator.name: interior_validator,
    }

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()
    # visualize prediction result
    pred_points = geom["rect"].sample_interior(NPOINT, evenly=True)
    pred_points["u"] = points["u"]
    pred_points["bc_data"] = np.zeros_like(points["bc_data"])
    pred_points["bc"] = np.full((NPOINT, 1), cfg.EVAL.bc_type, dtype="float32")
    pred = solver.predict(pred_points)
    logger.message("Now saving visual result to: visual/result.vtu, please wait...")
    ppsci.visualize.save_vtu_from_dict(
        osp.join(cfg.output_dir, "visual/result.vtu"),
        {
            "x": pred_points["x"],
            "y": pred_points["y"],
            "T": pred["T"],
        },
        (
            "x",
            "y",
        ),
        ("T"),
    )


@hydra.main(version_base=None, config_path="./conf", config_name="chip_heat.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

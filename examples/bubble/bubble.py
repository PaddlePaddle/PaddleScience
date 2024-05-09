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

"""
Reference: https://github.com/hanfengzhai/BubbleNet
Bubble data files download link: https://paddle-org.bj.bcebos.com/paddlescience/datasets/BubbleNet/bubble.mat
"""

from os import path as osp

import hydra
import numpy as np
import paddle
import scipy
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # load Data
    data = scipy.io.loadmat(cfg.DATA_PATH)
    # normalize data
    p_max = data["p"].max(axis=0)
    p_min = data["p"].min(axis=0)
    p_norm = (data["p"] - p_min) / (p_max - p_min)
    u_max = data["u"].max(axis=0)
    u_min = data["u"].min(axis=0)
    u_norm = (data["u"] - u_min) / (u_max - u_min)
    v_max = data["v"].max(axis=0)
    v_min = data["v"].min(axis=0)
    v_norm = (data["v"] - v_min) / (v_max - v_min)

    u_star = u_norm  # N x T
    v_star = v_norm  # N x T
    p_star = p_norm  # N x T
    phil_star = data["phil"]  # N x T
    t_star = data["t"]  # T x 1
    x_star = data["X"]  # N x 2

    N = x_star.shape[0]
    T = t_star.shape[0]

    # rearrange data
    xx = np.tile(x_star[:, 0:1], (1, T))  # N x T
    yy = np.tile(x_star[:, 1:2], (1, T))  # N x T
    tt = np.tile(t_star, (1, N)).T  # N x T

    x = xx.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    y = yy.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    t = tt.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1

    u = u_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    v = v_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    p = p_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    phil = phil_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1

    idx = np.random.choice(N * T, int(N * T * 0.75), replace=False)
    # train data
    train_input = {"x": x[idx, :], "y": y[idx, :], "t": t[idx, :]}
    train_label = {"u": u[idx, :], "v": v[idx, :], "p": p[idx, :], "phil": phil[idx, :]}

    # eval data
    test_input = {"x": x, "y": y, "t": t}
    test_label = {"u": u, "v": v, "p": p, "phil": phil}

    # set model
    model_psi = ppsci.arch.MLP(**cfg.MODEL.psi_net)
    model_p = ppsci.arch.MLP(**cfg.MODEL.p_net)
    model_phil = ppsci.arch.MLP(**cfg.MODEL.phil_net)

    # transform
    def transform_out(in_, out):
        psi_y = out["psi"]
        y = in_["y"]
        x = in_["x"]
        u = jacobian(psi_y, y)
        v = -jacobian(psi_y, x)
        return {"u": u, "v": v}

    # register transform
    model_psi.register_output_transform(transform_out)
    model_list = ppsci.arch.ModelList((model_psi, model_p, model_phil))

    # set time-geometry
    # set timestamps(including initial t0)
    timestamps = np.linspace(0, 126, 127, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.PointCloud(
            train_input,
            ("t", "x", "y"),
        ),
        "time_rect_visu": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(1, 126, timestamps=timestamps),
            ppsci.geometry.Rectangle((0, 0), (15, 5)),
        ),
    }

    NTIME_ALL = len(timestamps)
    NPOINT_PDE, NTIME_PDE = 300 * 100, NTIME_ALL - 1

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        {
            "pressure_Poisson": lambda out: hessian(out["p"], out["x"])
            + hessian(out["p"], out["y"])
        },
        {"pressure_Poisson": 0},
        geom["time_rect"],
        {
            "dataset": "IterableNamedArrayDataset",
            "batch_size": cfg.TRAIN.batch_size.pde_constraint,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": train_input,
                "label": train_label,
            },
            "batch_size": cfg.TRAIN.batch_size.sup_constraint,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="Sup",
    )

    # wrap constraints together
    constraint = {
        sup_constraint.name: sup_constraint,
        pde_constraint.name: pde_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model_list)

    # set validator
    mse_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_input,
                "label": test_label,
            },
            "batch_size": cfg.TRAIN.batch_size.mse_validator,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="bubble_mse",
    )
    validator = {
        mse_validator.name: mse_validator,
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        geom=geom,
        validator=validator,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # visualize prediction after finished training
    visu_mat = geom["time_rect_visu"].sample_interior(
        NPOINT_PDE * NTIME_PDE, evenly=True
    )
    # transform
    def transform_out(in_, out):
        psi_y = out["psi"]
        y = in_["y"]
        x = in_["x"]
        u = jacobian(psi_y, y, create_graph=False)
        v = -jacobian(psi_y, x, create_graph=False)
        return {"u": u, "v": v}

    model_psi.register_output_transform(transform_out)

    pred_norm = solver.predict(visu_mat, None, 4096, no_grad=False, return_numpy=True)
    # inverse normalization
    p_pred = pred_norm["p"].reshape([NTIME_PDE, NPOINT_PDE]).T
    u_pred = pred_norm["u"].reshape([NTIME_PDE, NPOINT_PDE]).T
    v_pred = pred_norm["v"].reshape([NTIME_PDE, NPOINT_PDE]).T
    pred = {
        "p": (p_pred * (p_max - p_min) + p_min).T.reshape([-1, 1]),
        "u": (u_pred * (u_max - u_min) + u_min).T.reshape([-1, 1]),
        "v": (v_pred * (v_max - v_min) + v_min).T.reshape([-1, 1]),
        "phil": pred_norm["phil"],
    }
    logger.message("Now saving visual result to: visual/result.vtu, please wait...")
    ppsci.visualize.save_vtu_from_dict(
        osp.join(cfg.output_dir, "visual/result.vtu"),
        {
            "t": visu_mat["t"],
            "x": visu_mat["x"],
            "y": visu_mat["y"],
            "u": pred["u"],
            "v": pred["v"],
            "p": pred["p"],
            "phil": pred["phil"],
        },
        ("t", "x", "y"),
        ("u", "v", "p", "phil"),
        NTIME_PDE,
    )


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # load Data
    data = scipy.io.loadmat(cfg.DATA_PATH)
    # normalize data
    p_max = data["p"].max(axis=0)
    p_min = data["p"].min(axis=0)
    p_norm = (data["p"] - p_min) / (p_max - p_min)
    u_max = data["u"].max(axis=0)
    u_min = data["u"].min(axis=0)
    u_norm = (data["u"] - u_min) / (u_max - u_min)
    v_max = data["v"].max(axis=0)
    v_min = data["v"].min(axis=0)
    v_norm = (data["v"] - v_min) / (v_max - v_min)

    u_star = u_norm  # N x T
    v_star = v_norm  # N x T
    p_star = p_norm  # N x T
    phil_star = data["phil"]  # N x T
    t_star = data["t"]  # T x 1
    x_star = data["X"]  # N x 2

    N = x_star.shape[0]
    T = t_star.shape[0]

    # rearrange data
    xx = np.tile(x_star[:, 0:1], (1, T))  # N x T
    yy = np.tile(x_star[:, 1:2], (1, T))  # N x T
    tt = np.tile(t_star, (1, N)).T  # N x T

    x = xx.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    y = yy.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    t = tt.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1

    u = u_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    v = v_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    p = p_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1
    phil = phil_star.flatten()[:, None].astype(paddle.get_default_dtype())  # NT x 1

    idx = np.random.choice(N * T, int(N * T * 0.75), replace=False)
    # train data
    train_input = {"x": x[idx, :], "y": y[idx, :], "t": t[idx, :]}

    # eval data
    test_input = {"x": x, "y": y, "t": t}
    test_label = {"u": u, "v": v, "p": p, "phil": phil}

    # set model
    model_psi = ppsci.arch.MLP(**cfg.MODEL.psi_net)
    model_p = ppsci.arch.MLP(**cfg.MODEL.p_net)
    model_phil = ppsci.arch.MLP(**cfg.MODEL.phil_net)

    # transform
    def transform_out(in_, out):
        psi_y = out["psi"]
        y = in_["y"]
        x = in_["x"]
        u = jacobian(psi_y, y, create_graph=False)
        v = -jacobian(psi_y, x, create_graph=False)
        return {"u": u, "v": v}

    # register transform
    model_psi.register_output_transform(transform_out)
    model_list = ppsci.arch.ModelList((model_psi, model_p, model_phil))

    # set time-geometry
    # set timestamps(including initial t0)
    timestamps = np.linspace(0, 126, 127, endpoint=True)
    geom = {
        "time_rect": ppsci.geometry.PointCloud(
            train_input,
            ("t", "x", "y"),
        ),
        "time_rect_visu": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(1, 126, timestamps=timestamps),
            ppsci.geometry.Rectangle((0, 0), (15, 5)),
        ),
    }

    NTIME_ALL = len(timestamps)
    NPOINT_PDE, NTIME_PDE = 300 * 100, NTIME_ALL - 1

    # set validator
    mse_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": test_input,
                "label": test_label,
            },
            "batch_size": cfg.TRAIN.batch_size.mse_validator,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="bubble_mse",
    )
    validator = {
        mse_validator.name: mse_validator,
    }

    # directly evaluate pretrained model(optional)
    solver = ppsci.solver.Solver(
        model_list,
        output_dir=cfg.output_dir,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()

    # visualize prediction
    visu_mat = geom["time_rect_visu"].sample_interior(
        NPOINT_PDE * NTIME_PDE, evenly=True
    )

    pred_norm = solver.predict(
        visu_mat, None, 4096 * 2, no_grad=False, return_numpy=True
    )
    # inverse normalization
    p_pred = pred_norm["p"].reshape([NTIME_PDE, NPOINT_PDE]).T
    u_pred = pred_norm["u"].reshape([NTIME_PDE, NPOINT_PDE]).T
    v_pred = pred_norm["v"].reshape([NTIME_PDE, NPOINT_PDE]).T
    pred = {
        "p": (p_pred * (p_max - p_min) + p_min).T.reshape([-1, 1]),
        "u": (u_pred * (u_max - u_min) + u_min).T.reshape([-1, 1]),
        "v": (v_pred * (v_max - v_min) + v_min).T.reshape([-1, 1]),
        "phil": pred_norm["phil"],
    }
    logger.message("Now saving visual result to: visual/result.vtu, please wait...")
    ppsci.visualize.save_vtu_from_dict(
        osp.join(cfg.output_dir, "visual/result.vtu"),
        {
            "t": visu_mat["t"],
            "x": visu_mat["x"],
            "y": visu_mat["y"],
            "u": pred["u"],
            "v": pred["v"],
            "p": pred["p"],
            "phil": pred["phil"],
        },
        ("t", "x", "y"),
        ("u", "v", "p", "phil"),
        NTIME_PDE,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="bubble.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

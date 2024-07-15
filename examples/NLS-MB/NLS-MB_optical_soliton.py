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

from os import path as osp

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def analytic_solution(out):
    t, x = out["t"], out["x"]
    Eu_true = 2 * np.cos(2 * t) / np.cosh(2 * t + 6 * x)

    Ev_true = -2 * np.sin(2 * t) / np.cosh(2 * t + 6 * x)

    pu_true = (
        (np.exp(-2 * t - 6 * x) - np.exp(2 * t + 6 * x))
        * np.cos(2 * t)
        / np.cosh(2 * t + 6 * x) ** 2
    )
    pv_true = (
        -(np.exp(-2 * t - 6 * x) - np.exp(2 * t + 6 * x))
        * np.sin(2 * t)
        / np.cosh(2 * t + 6 * x) ** 2
    )
    eta_true = (np.cosh(2 * t + 6 * x) ** 2 - 2) / np.cosh(2 * t + 6 * x) ** 2

    return Eu_true, Ev_true, pu_true, pv_true, eta_true


def plot(
    t: np.ndarray,
    x: np.ndarray,
    E_ref: np.ndarray,
    E_pred: np.ndarray,
    p_ref: np.ndarray,
    p_pred: np.ndarray,
    eta_ref: np.ndarray,
    eta_pred: np.ndarray,
    output_dir: str,
):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.title("E_ref")
    plt.tricontourf(x, t, E_ref, levels=256, cmap="jet")
    plt.subplot(3, 3, 2)
    plt.title("E_pred")
    plt.tricontourf(x, t, E_pred, levels=256, cmap="jet")
    plt.subplot(3, 3, 3)
    plt.title("E_diff")
    plt.tricontourf(x, t, np.abs(E_ref - E_pred), levels=256, cmap="jet")
    plt.subplot(3, 3, 4)
    plt.title("p_ref")
    plt.tricontourf(x, t, p_ref, levels=256, cmap="jet")
    plt.subplot(3, 3, 5)
    plt.title("p_pred")
    plt.tricontourf(x, t, p_pred, levels=256, cmap="jet")
    plt.subplot(3, 3, 6)
    plt.title("p_diff")
    plt.tricontourf(x, t, np.abs(p_ref - p_pred), levels=256, cmap="jet")
    plt.subplot(3, 3, 7)
    plt.title("eta_ref")
    plt.tricontourf(x, t, eta_ref, levels=256, cmap="jet")
    plt.subplot(3, 3, 8)
    plt.title("eta_pred")
    plt.tricontourf(x, t, eta_pred, levels=256, cmap="jet")
    plt.subplot(3, 3, 9)
    plt.title("eta_diff")
    plt.tricontourf(x, t, np.abs(eta_ref - eta_pred), levels=256, cmap="jet")
    fig_path = osp.join(output_dir, "pred_optical_soliton.png")
    print(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {
        "NLS-MB": ppsci.equation.NLSMB(alpha_1=0.5, alpha_2=-1, omega_0=-1, time=True)
    }

    x_lower = -1
    x_upper = 1
    t_lower = -1
    t_upper = 1
    # set timestamps(including initial t0)
    timestamps = np.linspace(t_lower, t_upper, cfg.NTIME_ALL, endpoint=True)
    # set time-geometry
    geom = {
        "time_interval": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(t_lower, t_upper, timestamps=timestamps),
            ppsci.geometry.Interval(x_lower, x_upper),
        )
    }

    X, T = np.meshgrid(
        np.linspace(x_lower, x_upper, 256), np.linspace(t_lower, t_upper, 256)
    )
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Boundary and Initial conditions
    ic = X_star[:, 1] == t_lower
    idx_ic = np.random.choice(np.where(ic)[0], 200, replace=False)
    lb = X_star[:, 0] == x_lower
    idx_lb = np.random.choice(np.where(lb)[0], 200, replace=False)
    ub = X_star[:, 0] == x_upper
    idx_ub = np.random.choice(np.where(ub)[0], 200, replace=False)
    icbc_idx = np.hstack((idx_lb, idx_ic, idx_ub))
    X_u_train = X_star[icbc_idx].astype("float32")
    X_u_train = {"t": X_u_train[:, 1:2], "x": X_u_train[:, 0:1]}

    Eu_train, Ev_train, pu_train, pv_train, eta_train = analytic_solution(X_u_train)

    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"t": X_u_train["t"], "x": X_u_train["x"]},
            "label": {
                "Eu": Eu_train,
                "Ev": Ev_train,
                "pu": pu_train,
                "pv": pv_train,
                "eta": eta_train,
            },
        },
        "batch_size": 600,
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    # set constraint
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NLS-MB"].equations,
        {
            "Schrodinger_1": 0,
            "Schrodinger_2": 0,
            "Maxwell_1": 0,
            "Maxwell_2": 0,
            "Bloch": 0,
        },
        geom["time_interval"],
        {
            "dataset": {"name": "IterableNamedArrayDataset"},
            "batch_size": 20000,
            "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        name="EQ",
    )

    # supervised constraint s.t ||u-u_0||
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="Sup",
    )

    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        sup_constraint.name: sup_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=cfg.TRAIN.learning_rate)(model)

    # set validator
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NLS-MB"].equations,
        {
            "Schrodinger_1": 0,
            "Schrodinger_2": 0,
            "Maxwell_1": 0,
            "Maxwell_2": 0,
            "Bloch": 0,
        },
        geom["time_interval"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": 20600,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
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

    # fine-tuning pretrained model with L-BFGS
    OUTPUT_DIR = cfg.TRAIN.lbfgs.output_dir
    logger.init_logger("ppsci", osp.join(OUTPUT_DIR, f"{cfg.mode}.log"), "info")
    EPOCHS = cfg.TRAIN.epochs // 10
    optimizer_lbfgs = ppsci.optimizer.LBFGS(
        cfg.TRAIN.lbfgs.learning_rate, cfg.TRAIN.lbfgs.max_iter
    )(model)
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer_lbfgs,
        None,
        EPOCHS,
        cfg.TRAIN.lbfgs.iters_per_epoch,
        eval_during_train=cfg.TRAIN.lbfgs.eval_during_train,
        eval_freq=cfg.TRAIN.lbfgs.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # visualize prediction
    vis_points = geom["time_interval"].sample_interior(20000, evenly=True)
    Eu_true, Ev_true, pu_true, pv_true, eta_true = analytic_solution(vis_points)
    pred = solver.predict(vis_points, return_numpy=True)
    t = vis_points["t"][:, 0]
    x = vis_points["x"][:, 0]
    E_ref = np.sqrt(Eu_true**2 + Ev_true**2)[:, 0]
    E_pred = np.sqrt(pred["Eu"] ** 2 + pred["Ev"] ** 2)[:, 0]
    p_ref = np.sqrt(pu_true**2 + pv_true**2)[:, 0]
    p_pred = np.sqrt(pred["pu"] ** 2 + pred["pv"] ** 2)[:, 0]
    eta_ref = eta_true[:, 0]
    eta_pred = pred["eta"][:, 0]

    # plot
    plot(t, x, E_ref, E_pred, p_ref, p_pred, eta_ref, eta_pred, cfg.output_dir)


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {
        "NLS-MB": ppsci.equation.NLSMB(alpha_1=0.5, alpha_2=-1, omega_0=-1, time=True)
    }

    # set geometry
    x_lower = -1
    x_upper = 1
    t_lower = -1
    t_upper = 1
    # set timestamps(including initial t0)
    timestamps = np.linspace(t_lower, t_upper, cfg.NTIME_ALL, endpoint=True)
    # set time-geometry
    geom = {
        "time_interval": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(t_lower, t_upper, timestamps=timestamps),
            ppsci.geometry.Interval(x_lower, x_upper),
        )
    }

    # set validator
    residual_validator = ppsci.validate.GeometryValidator(
        equation["NLS-MB"].equations,
        {
            "Schrodinger_1": 0,
            "Schrodinger_2": 0,
            "Maxwell_1": 0,
            "Maxwell_2": 0,
            "Bloch": 0,
        },
        geom["time_interval"],
        {
            "dataset": "IterableNamedArrayDataset",
            "total_size": 20600,
        },
        ppsci.loss.MSELoss(),
        evenly=True,
        metric={"MSE": ppsci.metric.MSE()},
        with_initial=True,
        name="Residual",
    )
    validator = {residual_validator.name: residual_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        eval_freq=cfg.TRAIN.eval_freq,
        equation=equation,
        geom=geom,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    solver.eval()

    # visualize prediction
    vis_points = geom["time_interval"].sample_interior(20000, evenly=True)
    Eu_true, Ev_true, pu_true, pv_true, eta_true = analytic_solution(vis_points)
    pred = solver.predict(vis_points, return_numpy=True)
    t = vis_points["t"][:, 0]
    x = vis_points["x"][:, 0]
    E_ref = np.sqrt(Eu_true**2 + Ev_true**2)[:, 0]
    E_pred = np.sqrt(pred["Eu"] ** 2 + pred["Ev"] ** 2)[:, 0]
    p_ref = np.sqrt(pu_true**2 + pv_true**2)[:, 0]
    p_pred = np.sqrt(pred["pu"] ** 2 + pred["pv"] ** 2)[:, 0]
    eta_ref = eta_true[:, 0]
    eta_pred = pred["eta"][:, 0]

    # plot
    plot(t, x, E_ref, E_pred, p_ref, p_pred, eta_ref, eta_pred, cfg.output_dir)


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

    # set geometry
    x_lower = -1
    x_upper = 1
    t_lower = -1
    t_upper = 1
    # set timestamps(including initial t0)
    timestamps = np.linspace(t_lower, t_upper, cfg.NTIME_ALL, endpoint=True)
    # set time-geometry
    geom = {
        "time_interval": ppsci.geometry.TimeXGeometry(
            ppsci.geometry.TimeDomain(t_lower, t_upper, timestamps=timestamps),
            ppsci.geometry.Interval(x_lower, x_upper),
        )
    }

    NPOINT_TOTAL = cfg.NPOINT_INTERIOR + cfg.NPOINT_BC
    input_dict = geom["time_interval"].sample_interior(NPOINT_TOTAL, evenly=True)

    output_dict = predictor.predict(
        {key: input_dict[key] for key in cfg.MODEL.input_keys}, cfg.INFER.batch_size
    )

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    # visualize prediction
    Eu_true, Ev_true, pu_true, pv_true, eta_true = analytic_solution(input_dict)
    t = input_dict["t"][:, 0]
    x = input_dict["x"][:, 0]
    E_ref = np.sqrt(Eu_true**2 + Ev_true**2)[:, 0]
    E_pred = np.sqrt(output_dict["Eu"] ** 2 + output_dict["Ev"] ** 2)[:, 0]
    p_ref = np.sqrt(pu_true**2 + pv_true**2)[:, 0]
    p_pred = np.sqrt(output_dict["pu"] ** 2 + output_dict["pv"] ** 2)[:, 0]
    eta_ref = eta_true[:, 0]
    eta_pred = output_dict["eta"][:, 0]

    # plot
    plot(t, x, E_ref, E_pred, p_ref, p_pred, eta_ref, eta_pred, cfg.output_dir)


@hydra.main(version_base=None, config_path="./conf", config_name="NLS-MB_soliton.yaml")
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

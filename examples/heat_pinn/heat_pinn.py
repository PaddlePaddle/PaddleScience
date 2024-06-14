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

import fdm
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def plot(input_data, N_EVAL, pinn_output, fdm_output, cfg):
    x = input_data["x"].reshape(N_EVAL, N_EVAL)
    y = input_data["y"].reshape(N_EVAL, N_EVAL)

    plt.subplot(2, 1, 1)
    plt.pcolormesh(x, y, pinn_output * 75.0, cmap="magma")
    plt.colorbar()
    plt.title("PINN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.axis("square")

    plt.subplot(2, 1, 2)
    plt.pcolormesh(x, y, fdm_output, cmap="magma")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("FDM")
    plt.tight_layout()
    plt.axis("square")
    plt.savefig(osp.join(cfg.output_dir, "pinn_fdm_comparison.png"))
    plt.close()

    frames_val = np.array([-0.75, -0.5, -0.25, 0.0, +0.25, +0.5, +0.75])
    frames = [*map(int, (frames_val + 1) / 2 * (N_EVAL - 1))]
    height = 3
    plt.figure("", figsize=(len(frames) * height, 2 * height))

    for i, var_index in enumerate(frames):
        plt.subplot(2, len(frames), i + 1)
        plt.title(f"y = {frames_val[i]:.2f}")
        plt.plot(
            x[:, var_index],
            pinn_output[:, var_index] * 75.0,
            "r--",
            lw=4.0,
            label="pinn",
        )
        plt.plot(x[:, var_index], fdm_output[:, var_index], "b", lw=2.0, label="FDM")
        plt.ylim(0.0, 100.0)
        plt.xlim(-1.0, +1.0)
        plt.xlabel("x")
        plt.ylabel("T")
        plt.tight_layout()
        plt.legend()

    for i, var_index in enumerate(frames):
        plt.subplot(2, len(frames), len(frames) + i + 1)
        plt.title(f"x = {frames_val[i]:.2f}")
        plt.plot(
            y[var_index, :],
            pinn_output[var_index, :] * 75.0,
            "r--",
            lw=4.0,
            label="pinn",
        )
        plt.plot(y[var_index, :], fdm_output[var_index, :], "b", lw=2.0, label="FDM")
        plt.ylim(0.0, 100.0)
        plt.xlim(-1.0, +1.0)
        plt.xlabel("y")
        plt.ylabel("T")
        plt.tight_layout()
        plt.legend()

    plt.savefig(osp.join(cfg.output_dir, "profiles.png"))


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set equation
    equation = {"heat": ppsci.equation.Laplace(dim=2)}

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-1.0, -1.0), (1.0, 1.0))}

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": cfg.TRAIN.iters_per_epoch,
    }

    # set constraint
    NPOINT_PDE = 99**2
    NPOINT_TOP = 25
    NPOINT_BOTTOM = 25
    NPOINT_LEFT = 25
    NPOINT_RIGHT = 25
    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["heat"].equations,
        {"laplace": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_PDE},
        ppsci.loss.MSELoss("mean"),
        evenly=True,
        name="EQ",
    )
    bc_top = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_TOP},
        ppsci.loss.MSELoss("mean"),
        weight_dict={"u": cfg.TRAIN.weight.bc_top},
        criteria=lambda x, y: np.isclose(y, 1),
        name="BC_top",
    )
    bc_bottom = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 50 / 75},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_BOTTOM},
        ppsci.loss.MSELoss("mean"),
        weight_dict={"u": cfg.TRAIN.weight.bc_bottom},
        criteria=lambda x, y: np.isclose(y, -1),
        name="BC_bottom",
    )
    bc_left = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 1},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_LEFT},
        ppsci.loss.MSELoss("mean"),
        weight_dict={"u": cfg.TRAIN.weight.bc_left},
        criteria=lambda x, y: np.isclose(x, -1),
        name="BC_left",
    )
    bc_right = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        geom["rect"],
        {**train_dataloader_cfg, "batch_size": NPOINT_RIGHT},
        ppsci.loss.MSELoss("mean"),
        weight_dict={"u": cfg.TRAIN.weight.bc_right},
        criteria=lambda x, y: np.isclose(x, 1),
        name="BC_right",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc_top.name: bc_top,
        bc_bottom.name: bc_bottom,
        bc_left.name: bc_left,
        bc_right.name: bc_right,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=cfg.TRAIN.learning_rate)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=cfg.TRAIN.iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )
    # train model
    solver.train()

    # begin eval
    N_EVAL = 100
    input_data = geom["rect"].sample_interior(N_EVAL**2, evenly=True)
    pinn_output = solver.predict(input_data, return_numpy=True)["u"].reshape(
        N_EVAL, N_EVAL
    )
    fdm_output = fdm.solve(N_EVAL, 1).T
    mse_loss = np.mean(np.square(pinn_output - (fdm_output / 75.0)))
    logger.info(f"The norm MSE loss between the FDM and PINN is {mse_loss}")
    plot(input_data, N_EVAL, pinn_output, fdm_output, cfg)


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # set output directory
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set geometry
    geom = {"rect": ppsci.geometry.Rectangle((-1.0, -1.0), (1.0, 1.0))}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )
    # begin eval
    N_EVAL = 100
    input_data = geom["rect"].sample_interior(N_EVAL**2, evenly=True)
    pinn_output = solver.predict(input_data, no_grad=True, return_numpy=True)[
        "u"
    ].reshape(N_EVAL, N_EVAL)
    fdm_output = fdm.solve(N_EVAL, 1).T
    mse_loss = np.mean(np.square(pinn_output - (fdm_output / 75.0)))
    logger.info(f"The norm MSE loss between the FDM and PINN is {mse_loss:.5e}")
    plot(input_data, N_EVAL, pinn_output, fdm_output, cfg)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
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
    geom = {"rect": ppsci.geometry.Rectangle((-1.0, -1.0), (1.0, 1.0))}
    # begin eval
    N_EVAL = 100
    input_data = geom["rect"].sample_interior(N_EVAL**2, evenly=True)
    output_data = predictor.predict(
        {key: input_data[key] for key in cfg.MODEL.input_keys}, cfg.INFER.batch_size
    )

    # mapping data to cfg.INFER.output_keys
    output_data = {
        store_key: output_data[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_data.keys())
    }["u"].reshape(N_EVAL, N_EVAL)
    fdm_output = fdm.solve(N_EVAL, 1).T
    mse_loss = np.mean(np.square(output_data - (fdm_output / 75.0)))
    logger.info(f"The norm MSE loss between the FDM and PINN is {mse_loss:.5e}")
    plot(input_data, N_EVAL, output_data, fdm_output, cfg)


@hydra.main(version_base=None, config_path="./conf", config_name="heat_pinn.yaml")
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

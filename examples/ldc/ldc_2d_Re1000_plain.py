"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/ldc
"""

from __future__ import annotations

import copy
import os
from os import path as osp

import hydra
import numpy as np
import paddle
import scipy.io as sio
from matplotlib import pyplot as plt
from omegaconf import DictConfig

import ppsci
from ppsci.utils import misc

dtype = paddle.get_default_dtype()


def plot(U_pred: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fig_path = osp.join(output_dir, "ac.png")

    fig = plt.figure()
    plt.pcolor(U_pred.T, cmap="jet")
    plt.colorbar()
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()
    ppsci.utils.logger.info(f"Saving figure to {fig_path}")


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    def sample_points_on_square_boundary(num_pts_per_side, eps):
        # Sample points along the top side (x=1 to x=0, y=1)
        top_coords = np.linspace(0, 1, num_pts_per_side)
        top = np.column_stack((top_coords, np.ones_like(top_coords)))

        # Sample points along the bottom side (x=0 to x=1, y=0)
        bottom_coords = np.linspace(0, 1, num_pts_per_side)
        bottom = np.column_stack((bottom_coords, np.zeros_like(bottom_coords)))

        # Sample points along the left side (x=0, y=1 to y=0)
        left_coords = np.linspace(0, 1 - eps, num_pts_per_side)
        left = np.column_stack((np.zeros_like(left_coords), left_coords))

        # Sample points along the right side (x=1, y=0 to y=1)
        right_coords = np.linspace(0, 1 - eps, num_pts_per_side)
        right = np.column_stack((np.ones_like(right_coords), right_coords))

        # Combine the points from all sides
        points = np.vstack((top, bottom, left, right))

        return points

    def train_curriculum(cfg, idx):
        cfg = copy.deepcopy(cfg)
        Re = cfg.Re[idx]
        cfg.output_dir = osp.join(cfg.output_dir, f"Re_{int(Re)}")
        cfg.TRAIN.epochs = cfg.epochs[idx]
        ppsci.utils.logger.message(
            f"Training curriculum {idx + 1}/{len(cfg.epochs)} Re={Re:.5g} epochs={cfg.epochs[idx]}"
        )

        # set equation
        equation = {
            "NavierStokes": ppsci.equation.NavierStokes(1 / Re, 1, dim=2, time=False)
        }

        # set constraint
        data = sio.loadmat(f"./data/ldc_Re{Re}.mat")
        u_ref = data["u"].astype(dtype)
        v_ref = data["v"].astype(dtype)
        U_ref = np.sqrt(u_ref**2 + v_ref**2).reshape(-1, 1)
        x_star = data["x"].flatten().astype(dtype)
        y_star = data["y"].flatten().astype(dtype)
        x0 = x_star[0]
        x1 = x_star[-1]
        y0 = y_star[0]
        y1 = y_star[-1]

        # set N-S pde constraint
        def gen_input_batch():
            tx = np.random.uniform(
                [x0, y0],
                [x1, y1],
                (cfg.TRAIN.batch_size.pde, 2),
            ).astype(dtype)
            return {"x": tx[:, 0:1], "y": tx[:, 1:2]}

        def gen_label_batch(input_batch):
            return {
                "continuity": np.zeros([cfg.TRAIN.batch_size.pde, 1], dtype),
                "momentum_x": np.zeros([cfg.TRAIN.batch_size.pde, 1], dtype),
                "momentum_y": np.zeros([cfg.TRAIN.batch_size.pde, 1], dtype),
            }

        pde_constraint = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": gen_input_batch,
                    "label": gen_label_batch,
                },
            },
            output_expr=equation["NavierStokes"].equations,
            loss=ppsci.loss.MSELoss("mean"),
            name="PDE",
        )

        # set boundary conditions
        x_bc = sample_points_on_square_boundary(
            cfg.TRAIN.batch_size.bc, eps=0.01
        ).astype(
            dtype
        )  # avoid singularity a right corner for u velocity
        v_bc = np.zeros((cfg.TRAIN.batch_size.bc * 4, 1), dtype)
        u_bc = copy.deepcopy(v_bc)
        u_bc[: cfg.TRAIN.batch_size.bc] = 1.0
        bc = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "IterableNamedArrayDataset",
                    "input": {
                        "x": x_bc[:, 0:1],
                        "y": x_bc[:, 1:2],
                    },
                    "label": {"u": u_bc, "v": v_bc},
                },
            },
            output_expr={"u": lambda out: out["u"], "v": lambda out: out["v"]},
            loss=ppsci.loss.MSELoss("mean"),
            name="BC",
        )
        # wrap constraints together
        constraint = {
            pde_constraint.name: pde_constraint,
            bc.name: bc,
        }

        # set validator
        xy_star = misc.cartesian_product(x_star, y_star).astype(dtype)
        eval_data = {"x": xy_star[:, 0:1], "y": xy_star[:, 1:2]}
        eval_label = {"U": U_ref.reshape([-1, 1])}
        U_validator = ppsci.validate.SupervisedValidator(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": eval_data,
                    "label": eval_label,
                },
                "batch_size": cfg.EVAL.batch_size,
            },
            ppsci.loss.MSELoss("mean"),
            {"U": lambda out: (out["u"] ** 2 + out["v"] ** 2).sqrt()},
            metric={"L2Rel": ppsci.metric.L2Rel()},
            name="U_validator",
        )
        validator = {U_validator.name: U_validator}

        # initialize solver
        solver = ppsci.solver.Solver(
            model,
            constraint,
            optimizer=optimizer,
            equation=equation,
            validator=validator,
            cfg=cfg,
        )
        # train model
        solver.train()
        # evaluate after finished training
        solver.eval()
        # visualize prediction after finished training
        pred_dict = solver.predict(
            eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
        )
        U_pred = np.sqrt(pred_dict["u"] ** 2 + pred_dict["v"] ** 2).reshape(
            [len(x_star), len(y_star)]
        )
        plot(U_pred, cfg.output_dir)

    for idx in range(len(cfg.Re)):
        train_curriculum(cfg, idx)


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    data = sio.loadmat(cfg.EVAL_DATA_PATH)
    data = dict(data)
    u_ref = data["u"].astype(dtype)
    v_ref = data["v"].astype(dtype)
    U_ref = np.sqrt(u_ref**2 + v_ref**2).reshape(-1, 1)
    x_star = data["x"].flatten().astype(dtype)  # [nx, ]
    y_star = data["y"].flatten().astype(dtype)  # [ny, ]

    # set validator
    xy_star = misc.cartesian_product(x_star, y_star).astype(dtype)
    eval_data = {"x": xy_star[:, 0:1], "y": xy_star[:, 1:2]}
    eval_label = {"U": U_ref.reshape([-1, 1])}
    U_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_label,
            },
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"U": lambda out: (out["u"] ** 2 + out["v"] ** 2).sqrt()},
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="U_validator",
    )
    validator = {U_validator.name: U_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )

    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    pred_dict = solver.predict(
        eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
    )
    U_pred = np.sqrt(pred_dict["u"] ** 2 + pred_dict["v"] ** 2).reshape(
        [len(x_star), len(y_star)]
    )
    # plot
    plot(U_pred, cfg.output_dir)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(model, cfg=cfg)
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {key: InputSpec([None, 1], "float32", name=key) for key in model.input_keys},
    ]
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)
    data = sio.loadmat(cfg.EVAL_DATA_PATH)
    t_star = data["t"].flatten().astype(dtype)  # [nt, ]
    x_star = data["x"].flatten().astype(dtype)  # [nx, ]
    y_star = data["y"].flatten().astype(dtype)  # [ny, ]
    tx_star = misc.cartesian_product(t_star, x_star).astype(dtype)

    input_dict = {"t": tx_star[:, 0:1], "x": tx_star[:, 1:2]}
    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }
    U_pred = np.sqrt(output_dict["u"] ** 2 + output_dict["v"] ** 2).reshape(
        [len(x_star), len(y_star)]
    )
    plot(U_pred, cfg.output_dir)


@hydra.main(
    version_base=None, config_path="./conf", config_name="ldc_2d_Re1000_plain.yaml"
)
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

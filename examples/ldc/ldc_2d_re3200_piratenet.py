"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/allen_cahn
"""

from __future__ import annotations

import copy
from os import path as osp
from typing import Optional
from typing import Tuple
from typing import Union

import hydra
import numpy as np
import paddle
import scipy.io as sio
import sympy as sp
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sympy.parsing import sympy_parser as sp_parser

import ppsci
from ppsci.equation.pde import base
from ppsci.loss import mtl
from ppsci.utils import misc

dtype = paddle.get_default_dtype()


class NS(base.PDE):
    def __init__(
        self,
        nu: Union[float, str],
        rho: Union[float, str],
        dim: int,
        time: bool,
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys
        self.dim = dim
        self.time = time

        t, x, y, z = self.create_symbols("t x y z")
        invars = (x, y)
        if time:
            invars = (t,) + invars
        if dim == 3:
            invars += (z,)

        if isinstance(nu, str):
            nu = sp_parser.parse_expr(nu)
            if isinstance(nu, sp.Symbol):
                invars += (nu,)

        if isinstance(rho, str):
            rho = sp_parser.parse_expr(rho)
            if isinstance(rho, sp.Symbol):
                invars += (rho,)

        self.nu = nu
        self.rho = rho

        u = self.create_function("u", invars)
        v = self.create_function("v", invars)
        w = self.create_function("w", invars) if dim == 3 else sp.Number(0)
        p = self.create_function("p", invars)

        continuity = u.diff(x) + v.diff(y) + w.diff(z)
        momentum_x = (
            u.diff(t)
            + u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)
            - (
                (nu * u.diff(x)).diff(x)
                + (nu * u.diff(y)).diff(y)
                + (nu * u.diff(z)).diff(z)
            )
            + 1 / rho * p.diff(x)
        )
        momentum_y = (
            v.diff(t)
            + u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)
            - (
                (nu * v.diff(x)).diff(x)
                + (nu * v.diff(y)).diff(y)
                + (nu * v.diff(z)).diff(z)
            )
            + 1 / rho * p.diff(y)
        )
        momentum_z = (
            w.diff(t)
            + u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            - (
                (nu * w.diff(x)).diff(x)
                + (nu * w.diff(y)).diff(y)
                + (nu * w.diff(z)).diff(z)
            )
            + 1 / rho * p.diff(z)
        )
        self.add_equation("continuity", continuity)
        self.add_equation("momentum_x", momentum_x)
        self.add_equation("momentum_y", momentum_y)
        if self.dim == 3:
            self.add_equation("momentum_z", momentum_z)

        self._apply_detach()


def plot(
    U_pred: np.ndarray,
    output_dir: str,
):
    fig = plt.figure()
    plt.pcolor(U_pred.T, cmap="jet")
    import os

    os.makedirs(output_dir, exist_ok=True)
    fig_path = osp.join(output_dir, "ac.png")
    print(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.PirateNet(**cfg.MODEL)
    grad_norm = mtl.GradNorm(
        model,
        3 + 2,  # 3pde + 2bc
        cfg.TRAIN.grad_norm.update_freq,
        cfg.TRAIN.grad_norm.momentum,
    )
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
        ppsci.utils.logger.message(f"Training curriculum {idx + 1}/{len(cfg.epochs)}")
        cfg = copy.deepcopy(cfg)
        cfg.TRAIN.epochs = cfg.epochs[idx]
        cfg.TRAIN.lr_scheduler.epochs = cfg.epochs[idx]
        Re = cfg.Re[idx]

        # set equation
        equation = {"NavierStokes": NS(1 / Re, 1, dim=2, time=False)}

        # set constraint
        data = sio.loadmat(f"./data/ldc_Re{Re}.mat")
        u_ref = data["u"].astype(dtype)
        print(f"u_ref.shape = {u_ref.shape} {u_ref.min()} ~ {u_ref.max()}")
        v_ref = data["v"].astype(dtype)
        print(f"v_ref.shape = {v_ref.shape} {v_ref.min()} ~ {v_ref.max()}")
        U_ref = np.sqrt(u_ref**2 + v_ref**2)
        print(f"U_ref.shape = {U_ref.shape} {U_ref.min()} ~ {U_ref.max()}")
        x_star = data["x"].flatten().astype(dtype)
        print(f"x_star.shape = {x_star.shape} {x_star.min()} ~ {x_star.max()}")
        y_star = data["y"].flatten().astype(dtype)
        print(f"y_star.shape = {y_star.shape} {y_star.min()} ~ {y_star.max()}")
        x0 = x_star[0]
        print(f"x0 = {x0}")
        x1 = x_star[-1]
        print(f"x1 = {x1}")

        y0 = y_star[0]
        print(f"y0 = {y0}")
        y1 = y_star[-1]
        print(f"y1 = {y1}")

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
        x_bc1 = sample_points_on_square_boundary(
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
                        "x": x_bc1[:, 0:1],
                        "y": x_bc1[:, 1:2],
                    },
                    "label": {"u": u_bc, "v": v_bc},
                },
            },
            output_expr={"u": lambda out: out["u"], "v": lambda out: out["u"]},
            loss=ppsci.loss.MSELoss("mean"),
            name="BC",
        )
        # wrap constraints together
        constraint = {
            pde_constraint.name: pde_constraint,
            bc.name: bc,
        }

        # set validator
        eval_data = {"x": x_star.reshape([-1, 1]), "y": y_star.reshape([-1, 1])}
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
            loss_aggregator=grad_norm,
            cfg=cfg,
        )
        solver.eval()
        # train model
        solver.train()
        # evaluate after finished training
        solver.eval()
        # visualize prediction after finished training
        pred_dict = solver.predict(
            eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
        )
        U_pred = np.sqrt(pred_dict["u"] ** 2 + pred_dict["v"] ** 2)
        plot(U_pred, cfg.output_dir)

    for idx in range(len(cfg.Re)):
        train_curriculum(cfg, idx)


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.PirateNet(**cfg.MODEL)

    data = sio.loadmat(cfg.DATA_PATH)
    u_ref = data["usol"].astype(dtype)  # (nt, nx)
    t_star = data["t"].flatten().astype(dtype)  # [nt, ]
    x_star = data["x"].flatten().astype(dtype)  # [nx, ]

    # set validator
    tx_star = misc.cartesian_product(t_star, x_star).astype(dtype)
    eval_data = {"t": tx_star[:, 0:1], "x": tx_star[:, 1:2]}
    eval_label = {"u": u_ref.reshape([-1, 1])}
    u_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": eval_data,
                "label": eval_label,
            },
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"u": lambda out: out["u"]},
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="u_validator",
    )
    validator = {u_validator.name: u_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )

    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    u_pred = solver.predict(
        eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
    )["u"]
    u_pred = u_pred.reshape([len(t_star), len(x_star)])

    # plot
    plot(t_star, x_star, u_ref, u_pred, cfg.output_dir)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.PirateNet(**cfg.MODEL)

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
    data = sio.loadmat(cfg.DATA_PATH)
    u_ref = data["usol"].astype(dtype)  # (nt, nx)
    t_star = data["t"].flatten().astype(dtype)  # [nt, ]
    x_star = data["x"].flatten().astype(dtype)  # [nx, ]
    tx_star = misc.cartesian_product(t_star, x_star).astype(dtype)

    input_dict = {"t": tx_star[:, 0:1], "x": tx_star[:, 1:2]}
    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }
    u_pred = output_dict["u"].reshape([len(t_star), len(x_star)])

    plot(t_star, x_star, u_ref, u_pred, cfg.output_dir)


@hydra.main(
    version_base=None, config_path="./conf", config_name="ldc_2d_re3200_piratenet.yaml"
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

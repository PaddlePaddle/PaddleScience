"""
Reference: https://github.com/stnamjef/SPINN/blob/main/helmholtz3d.py
"""

from os import path as osp

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger

dtype = paddle.get_default_dtype()


def save_result(filename, x, y, z, u_pred, u_ref):
    xm, ym, zm = np.meshgrid(x, y, z, indexing="ij")
    xm = xm.reshape(-1, 1)
    ym = ym.reshape(-1, 1)
    zm = zm.reshape(-1, 1)
    u_pred = u_pred.reshape(-1, 1)
    u_ref = u_ref.reshape(-1, 1)
    ppsci.visualize.save_vtu_from_dict(
        filename,
        {
            "x": xm,
            "y": ym,
            "z": zm,
            "u_pred": u_pred,
            "u_ref": u_ref,
        },
        ("x", "y", "z"),
        ("u_pred", "u_ref"),
    )


def _helmholtz3d_exact_u(a1, a2, a3, x, y, z):
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y) * np.sin(a3 * np.pi * z)


def _helmholtz3d_source_term(a1, a2, a3, x, y, z, lda=1.0):
    u_gt = _helmholtz3d_exact_u(a1, a2, a3, x, y, z)[..., None]
    uxx = -((a1 * np.pi) ** 2) * u_gt
    uyy = -((a2 * np.pi) ** 2) * u_gt
    uzz = -((a3 * np.pi) ** 2) * u_gt
    return uxx + uyy + uzz + lda * u_gt


def generate_train_helmholtz3d(a1, a2, a3, nc):
    xc = np.random.uniform(-1.0, 1.0, [nc, 1]).astype(dtype)
    yc = np.random.uniform(-1.0, 1.0, [nc, 1]).astype(dtype)
    zc = np.random.uniform(-1.0, 1.0, [nc, 1]).astype(dtype)
    # source term
    xcm, ycm, zcm = np.meshgrid(xc, yc, zc, indexing="ij")
    uc = _helmholtz3d_source_term(a1, a2, a3, xcm, ycm, zcm).astype(dtype)
    # boundary (hard-coded)
    xb = [
        np.asarray([[1.0]], dtype=dtype),
        np.asarray([[-1.0]], dtype=dtype),
        xc,
        xc,
        xc,
        xc,
    ]
    yb = [
        yc,
        yc,
        np.asarray([[1.0]], dtype=dtype),
        np.asarray([[-1.0]], dtype=dtype),
        yc,
        yc,
    ]
    zb = [
        zc,
        zc,
        zc,
        zc,
        np.asarray([[1.0]], dtype=dtype),
        np.asarray([[-1.0]], dtype=dtype),
    ]
    return xc, yc, zc, uc, xb, yb, zb


def generate_test_helmholtz3d(a1, a2, a3, nc_test):
    x = np.linspace(-1.0, 1.0, nc_test, dtype=dtype)
    y = np.linspace(-1.0, 1.0, nc_test, dtype=dtype)
    z = np.linspace(-1.0, 1.0, nc_test, dtype=dtype)
    xm, ym, zm = np.meshgrid(x, y, z, indexing="ij")
    u_gt = _helmholtz3d_exact_u(a1, a2, a3, xm, ym, zm).astype(dtype)[..., None]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    return x, y, z, u_gt


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.SPINN(**cfg.MODEL)

    # set equation
    equation = {"Helmholtz": ppsci.equation.Helmholtz(3, 1.0)}
    equation["Helmholtz"].model = model  # set model to equation for hvp

    # set constraint
    class InteriorDataGenerator:
        def __init__(self):
            self.iter = 0
            self._gen()

        def _gen(self):
            global xb, yb, zb
            xc, yc, zc, uc, xb, yb, zb = generate_train_helmholtz3d(
                cfg.a1,
                cfg.a2,
                cfg.a3,
                cfg.TRAIN.nc,
            )
            self.xc = xc
            self.yc = yc
            self.zc = zc
            self.uc = uc

        def __call__(self):
            self.iter += 1

            if self.iter % 100 == 0:
                self._gen()

            return {
                "x": self.xc,
                "y": self.yc,
                "z": self.zc,
                "uc": self.uc,
            }

    def gen_label_batch(input_batch):
        return {"helmholtz": input_batch["uc"]}

    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": InteriorDataGenerator(),
                "label": gen_label_batch,
            },
        },
        output_expr=equation["Helmholtz"].equations,
        loss=ppsci.loss.MSELoss("mean"),
        name="PDE",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
    }

    class BCDataGenerator:
        def __init__(self, idx: int):
            self.idx = idx

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[self.idx],
                "y": yb[self.idx],
                "z": zb[self.idx],
            }
            return tmp

    def gen_bc_label(data_dict):
        nx = len(data_dict["x"])
        ny = len(data_dict["y"])
        nz = len(data_dict["z"])
        return {"u": np.zeros([nx, ny, nz, 1])}

    for i in range(6):
        bc_constraint_i = ppsci.constraint.SupervisedConstraint(
            {
                "dataset": {
                    "name": "ContinuousNamedArrayDataset",
                    "input": BCDataGenerator(i),
                    "label": gen_bc_label,
                },
            },
            output_expr={"u": lambda out: out["u"]},
            loss=ppsci.loss.MSELoss("mean"),
            name=f"BC{i}",
        )
        constraint[bc_constraint_i.name] = bc_constraint_i

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        equation=equation,
        cfg=cfg,
    )
    # train model
    solver.train()

    # evaluate after training
    x, y, z, u_gt = generate_test_helmholtz3d(cfg.a1, cfg.a2, cfg.a3, cfg.EVAL.nc)
    u_pred = solver.predict(
        {
            "x": x,
            "y": y,
            "z": z,
        },
        batch_size=None,
        return_numpy=True,
    )["u"].reshape(-1)
    u_gt = u_gt.reshape(-1)
    l2_err = np.linalg.norm(u_pred - u_gt, ord=2) / np.linalg.norm(u_gt, ord=2)
    rmse = np.sqrt(np.mean((u_pred - u_gt) ** 2))
    logger.message(f"l2_err = {l2_err:.4f}, rmse = {rmse:.4f}")

    save_result(
        osp.join(cfg.output_dir, "helmholtz3d_result.vtu"), x, y, z, u_pred, u_gt
    )


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.SPINN(**cfg.MODEL)

    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
    )

    # evaluate
    x, y, z, u_gt = generate_test_helmholtz3d(cfg.a1, cfg.a2, cfg.a3, cfg.EVAL.nc)
    u_pred = solver.predict(
        {
            "x": x,
            "y": y,
            "z": z,
        },
        batch_size=None,
        return_numpy=True,
    )["u"].reshape(-1)
    u_gt = u_gt.reshape(-1)
    l2_err = np.linalg.norm(u_pred - u_gt, ord=2) / np.linalg.norm(u_gt, ord=2)
    rmse = np.sqrt(np.mean((u_pred - u_gt) ** 2))
    logger.message(f"l2_err = {l2_err:.4f}, rmse = {rmse:.4f}")

    save_result(
        osp.join(cfg.output_dir, "helmholtz3d_result.vtu"), x, y, z, u_pred, u_gt
    )


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.SPINN(**cfg.MODEL)

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

    # evaluate
    x, y, z, u_gt = generate_test_helmholtz3d(cfg.a1, cfg.a2, cfg.a3, cfg.EVAL.nc)
    output_dict = predictor.predict(
        {
            "x": x,
            "y": y,
            "z": z,
        },
        batch_size=None,
    )
    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }
    u_pred = output_dict["u"].reshape(-1)
    u_gt = u_gt.reshape(-1)
    l2_err = np.linalg.norm(u_pred - u_gt, ord=2) / np.linalg.norm(u_gt, ord=2)
    rmse = np.sqrt(np.mean((u_pred - u_gt) ** 2))
    logger.message(f"l2_err = {l2_err:.4f}, rmse = {rmse:.4f}")

    save_result(
        osp.join(cfg.output_dir, "helmholtz3d_result.vtu"), x, y, z, u_pred, u_gt
    )


@hydra.main(version_base=None, config_path="./conf", config_name="helmholtz3d.yaml")
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

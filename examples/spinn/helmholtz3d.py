"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/allen_cahn
"""

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


def plot(
    t_star: np.ndarray,
    x_star: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    output_dir: str,
):
    fig = plt.figure(figsize=(18, 5))
    TT, XX = np.meshgrid(t_star, x_star, indexing="ij")
    u_ref = u_ref.reshape([len(t_star), len(x_star)])

    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, np.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    fig_path = osp.join(output_dir, "ac.png")
    print(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()


def helmholtz3d_exact_u(a1, a2, a3, x, y, z):
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y) * np.sin(a3 * np.pi * z)


def helmholtz3d_source_term(a1, a2, a3, x, y, z, lda=1.0):
    u_gt = helmholtz3d_exact_u(a1, a2, a3, x, y, z)
    uxx = -((a1 * np.pi) ** 2) * u_gt
    uyy = -((a2 * np.pi) ** 2) * u_gt
    uzz = -((a3 * np.pi) ** 2) * u_gt
    return uxx + uyy + uzz + lda * u_gt


def _spinn_train_generator_helmholtz3d(a1, a2, a3, nc):
    # collocation points
    xc = np.random.uniform(-1.0, 1.0, [nc]).astype("float32")
    yc = np.random.uniform(-1.0, 1.0, [nc]).astype("float32")
    zc = np.random.uniform(-1.0, 1.0, [nc]).astype("float32")
    # source term
    xcm, ycm, zcm = np.meshgrid(xc, yc, zc, indexing="ij")
    uc = helmholtz3d_source_term(a1, a2, a3, xcm, ycm, zcm).astype("float32")
    # xc, yc, zc = xc.reshape(-1, 1), yc.reshape(-1, 1), zc.reshape(-1, 1)
    # uc = uc.reshape(-1, 1)
    # boundary (hard-coded)
    xb = [
        np.asarray([1.0], dtype="float32"),
        np.asarray([-1.0], dtype="float32"),
        xc,
        xc,
        xc,
        xc,
    ]
    yb = [
        yc,
        yc,
        np.asarray([1.0], dtype="float32"),
        np.asarray([-1.0], dtype="float32"),
        yc,
        yc,
    ]
    zb = [
        zc,
        zc,
        zc,
        zc,
        np.asarray([1.0], dtype="float32"),
        np.asarray([-1.0], dtype="float32"),
    ]
    return xc, yc, zc, uc, xb, yb, zb


def _test_generator_helmholtz3d(a1, a2, a3, nc_test):
    x = np.linspace(-1.0, 1.0, nc_test, dtype="float32")
    y = np.linspace(-1.0, 1.0, nc_test, dtype="float32")
    z = np.linspace(-1.0, 1.0, nc_test, dtype="float32")
    xm, ym, zm = np.meshgrid(x, y, z, indexing="ij")
    u_gt = helmholtz3d_exact_u(a1, a2, a3, xm, ym, zm).astype("float32")
    # u_gt = u_gt.reshape(-1, 1)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    return x, y, z, u_gt


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.SPINN(**cfg.MODEL)

    # set equation
    equation = {"Helmholtz": ppsci.equation.Helmholtz(3, 1.0, "uc")}
    equation["Helmholtz"].model = model

    # set constraint
    class pde_sample:
        def __init__(self):
            self.iter = 0
            self._gen()

        def _gen(self):
            global xb, yb, zb
            xc, yc, zc, uc, xb, yb, zb = _spinn_train_generator_helmholtz3d(
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

            tmp = {
                "x": self.xc,
                "y": self.yc,
                "z": self.zc,
                "uc": self.uc,
            }

            return tmp

    class bc_sample1:
        def __init__(self):
            pass

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[0],
                "y": yb[0],
                "z": zb[0],
            }
            return tmp

    class bc_sample2:
        def __init__(self):
            pass

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[1],
                "y": yb[1],
                "z": zb[1],
            }
            return tmp

    class bc_sample3:
        def __init__(self):
            pass

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[2],
                "y": yb[2],
                "z": zb[2],
            }
            return tmp

    class bc_sample4:
        def __init__(self):
            pass

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[3],
                "y": yb[3],
                "z": zb[3],
            }
            return tmp

    class bc_sample5:
        def __init__(self):
            pass

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[4],
                "y": yb[4],
                "z": zb[4],
            }
            return tmp

    class bc_sample6:
        def __init__(self):
            pass

        def __call__(self):
            global xb, yb, zb
            tmp = {
                "x": xb[5],
                "y": yb[5],
                "z": zb[5],
            }
            return tmp

    def gen_label_batch(input_batch):
        return {"helmholtz": input_batch["uc"]}

    def gen_label_batch_bc(data_dict):
        N = len(data_dict["x"]) * len(data_dict["y"]) * len(data_dict["z"])
        return {"u": np.zeros([N, 1])}

    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": pde_sample(),
                "label": gen_label_batch,
            },
        },
        output_expr=equation["Helmholtz"].equations,
        loss=ppsci.loss.MSELoss("mean"),
        name="PDE",
    )
    bc1_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": bc_sample1(),
                "label": gen_label_batch_bc,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="BC1",
    )
    bc2_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": bc_sample2(),
                "label": gen_label_batch_bc,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="BC2",
    )
    bc3_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": bc_sample3(),
                "label": gen_label_batch_bc,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="BC3",
    )
    bc4_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": bc_sample4(),
                "label": gen_label_batch_bc,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="BC4",
    )
    bc5_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": bc_sample5(),
                "label": gen_label_batch_bc,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="BC5",
    )
    bc6_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": bc_sample6(),
                "label": gen_label_batch_bc,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="BC6",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        bc1_constraint.name: bc1_constraint,
        bc2_constraint.name: bc2_constraint,
        bc3_constraint.name: bc3_constraint,
        bc4_constraint.name: bc4_constraint,
        bc5_constraint.name: bc5_constraint,
        bc6_constraint.name: bc6_constraint,
    }

    # set optimizer
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    # set validator
    # u_validator = ppsci.validate.SupervisedValidator(
    #     {
    #         "dataset": {
    #             "name": "NamedArrayDataset",
    #             "input": {'x': x, 'y': y, 'z': z},
    #             "label": {'u': u_gt},
    #         },
    #         "batch_size": cfg.EVAL.batch_size,
    #     },
    #     ppsci.loss.MSELoss("mean"),
    #     metric={"L2Rel": ppsci.metric.L2Rel()},
    #     name="u_validator",
    # )
    # validator = {u_validator.name: u_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        equation=equation,
        # validator=validator,
        cfg=cfg,
    )
    # train model
    solver.train()

    def compute_l2_error():
        x, y, z, u_gt = _test_generator_helmholtz3d(cfg.a1, cfg.a2, cfg.a3, cfg.EVAL.nc)
        u_pred = solver.predict(
            {
                "x": x,
                "y": y,
                "z": z,
            },
            batch_size=None,
            return_numpy=True,
        )["u"]
        l2_err = np.linalg.norm(u_pred - u_gt, ord="fro") / np.linalg.norm(
            u_gt, ord="fro"
        )
        print(f"l2_err = {l2_err:.3f}")

    compute_l2_error()
    # evaluate after finished training
    # solver.eval()
    # visualize prediction after finished training
    # u_pred = solver.predict(
    #     eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
    # )["u"]
    # u_pred = u_pred.reshape([len(t_star), len(x_star)])

    # # plot
    # plot(t_star, x_star, u_ref, u_pred, cfg.output_dir)


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

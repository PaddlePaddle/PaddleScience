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
from ppsci.loss import mtl
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


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.PirateNet(**cfg.MODEL)

    # set equation
    equation = {"AllenCahn": ppsci.equation.AllenCahn(eps=0.01)}

    # set constraint
    data = sio.loadmat(cfg.DATA_PATH)
    u_ref = data["usol"].astype(dtype)  # (nt, nx)
    t_star = data["t"].flatten().astype(dtype)  # [nt, ]
    x_star = data["x"].flatten().astype(dtype)  # [nx, ]

    u0 = u_ref[0, :]  # [nx, ]

    t0 = t_star[0]  # float
    t1 = t_star[-1]  # float

    x0 = x_star[0]  # float
    x1 = x_star[-1]  # float

    def gen_input_batch():
        tx = np.random.uniform(
            [t0, x0],
            [t1, x1],
            (cfg.TRAIN.batch_size, 2),
        ).astype(dtype)
        return {
            "t": np.sort(tx[:, 0:1], axis=0),
            "x": tx[:, 1:2],
        }

    def gen_label_batch(input_batch):
        return {"allen_cahn": np.zeros([cfg.TRAIN.batch_size, 1], dtype)}

    pde_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": gen_input_batch,
                "label": gen_label_batch,
            },
        },
        output_expr=equation["AllenCahn"].equations,
        loss=ppsci.loss.CausalMSELoss(
            cfg.TRAIN.causal.n_chunks, "mean", tol=cfg.TRAIN.causal.tol
        ),
        name="PDE",
    )

    ic_input = {"t": np.full([len(x_star), 1], t0), "x": x_star.reshape([-1, 1])}
    ic_label = {"u": u0.reshape([-1, 1])}
    ic = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableNamedArrayDataset",
                "input": ic_input,
                "label": ic_label,
            },
        },
        output_expr={"u": lambda out: out["u"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="IC",
    )
    # wrap constraints together
    constraint = {
        pde_constraint.name: pde_constraint,
        ic.name: ic,
    }

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

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
        constraint,
        optimizer=optimizer,
        equation=equation,
        validator=validator,
        loss_aggregator=mtl.GradNorm(
            model,
            len(constraint),
            cfg.TRAIN.grad_norm.update_freq,
            cfg.TRAIN.grad_norm.momentum,
        ),
        cfg=cfg,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()
    # visualize prediction after finished training
    u_pred = solver.predict(
        eval_data, batch_size=cfg.EVAL.batch_size, return_numpy=True
    )["u"]
    u_pred = u_pred.reshape([len(t_star), len(x_star)])

    # plot
    plot(t_star, x_star, u_ref, u_pred, cfg.output_dir)


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
    version_base=None, config_path="./conf", config_name="allen_cahn_piratenet.yaml"
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

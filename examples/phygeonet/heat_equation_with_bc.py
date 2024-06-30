from os import path as osp
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
import utils
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    model = ppsci.arch.USCNN(**cfg.MODEL)
    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

    data = np.load(cfg.data_dir)
    coords = data["coords"]
    jinvs = data["jinvs"]
    dxdxis = data["dxdxis"]
    dydxis = data["dydxis"]
    dxdetas = data["dxdetas"]
    dydetas = data["dydetas"]

    iters_per_epoch = coords.shape[0]
    sup_constraint_res = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "coords": coords,
                    "jinvs": jinvs,
                    "dxdxis": dxdxis,
                    "dydxis": dydxis,
                    "dxdetas": dxdetas,
                    "dydetas": dydetas,
                },
            },
            "batch_size": cfg.TRAIN.batch_size,
            "iters_per_epoch": iters_per_epoch,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(
            lambda out, label, weight: {"residual": out["residual"]}
        ),
        name="residual",
    )
    sup_constraint = {sup_constraint_res.name: sup_constraint_res}

    def _transform_out(
        _input: Dict[str, paddle.Tensor],
        _output: Dict[str, paddle.Tensor],
        pad_singleside: int = cfg.MODEL.pad_singleside,
    ):
        """Calculation residual.

        Args:
            _input (Dict[str, paddle.Tensor]): The input of the model.
            _output (Dict[str, paddle.Tensor]): The output of the model.
            pad_singleside (int, optional): Pad size. Defaults to cfg.MODEL.pad_singleside.
        """
        output_v = _output["output_v"]
        batch_size = output_v.shape[0]
        jinv = _input["jinvs"]
        dxdxi = _input["dxdxis"]
        dydxi = _input["dydxis"]
        dxdeta = _input["dxdetas"]
        dydeta = _input["dydetas"]
        Para = _input["coords"]
        for j in range(batch_size):
            output_v[j, 0, -pad_singleside:, pad_singleside:-pad_singleside] = output_v[
                j, 0, 1:2, pad_singleside:-pad_singleside
            ]
            output_v[j, 0, :pad_singleside, pad_singleside:-pad_singleside] = output_v[
                j, 0, -2:-1, pad_singleside:-pad_singleside
            ]
            output_v[j, 0, :, -pad_singleside:] = 0
            output_v[j, 0, :, 0:pad_singleside] = Para[j, 0, 0, 0]
        dvdx = utils.dfdx(output_v, dydeta, dydxi, jinv)
        d2vdx2 = utils.dfdx(dvdx, dydeta, dydxi, jinv)
        dvdy = utils.dfdy(output_v, dxdxi, dxdeta, jinv)
        d2vdy2 = utils.dfdy(dvdy, dxdxi, dxdeta, jinv)
        continuity = d2vdy2 + d2vdx2
        return {"residual": paddle.mean(continuity**2)}

    model.register_output_transform(_transform_out)
    solver = ppsci.solver.Solver(
        model,
        sup_constraint,
        optimizer=optimizer,
        cfg=cfg,
    )

    solver.train()
    solver.plot_loss_history()


def evaluate(cfg: DictConfig):
    pad_singleside = cfg.MODEL.pad_singleside
    model = ppsci.arch.USCNN(**cfg.MODEL)

    data = np.load(cfg.test_data_dir)
    paras = paddle.to_tensor(data["paras"])
    truths = paddle.to_tensor(data["truths"])
    coords = paddle.to_tensor(data["coords"])
    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
    )

    paras = paras.reshape([paras.shape[0], 1, paras.shape[1], paras.shape[2]])
    output = solver.predict({"coords": paras})
    output_v = output["output_v"]
    num_sample = output_v.shape[0]
    for j in range(num_sample):
        # Impose BC
        output_v[j, 0, -pad_singleside:, pad_singleside:-pad_singleside] = output_v[
            j, 0, 1:2, pad_singleside:-pad_singleside
        ]
        output_v[j, 0, :pad_singleside, pad_singleside:-pad_singleside] = output_v[
            j, 0, -2:-1, pad_singleside:-pad_singleside
        ]
        output_v[j, 0, :, -pad_singleside:] = 0
        output_v[j, 0, :, 0:pad_singleside] = paras[j, 0, 0, 0]

    error = paddle.sqrt(
        paddle.mean((truths - output_v) ** 2) / paddle.mean(truths**2)
    ).item()
    logger.info(f"The average error: {error / num_sample}")
    output_vs = output_v.numpy()
    PARALIST = [1, 2, 3, 4, 5, 6, 7]
    for i in range(len(PARALIST)):
        truth = truths[i].numpy()
        coord = coords[i].numpy()
        output_v = output_vs[i]
        truth = truth.reshape(1, 1, truth.shape[0], truth.shape[1])
        coord = coord.reshape(1, 2, coord.shape[2], coord.shape[3])
        fig1 = plt.figure()
        xylabelsize = 20
        xytickssize = 20
        titlesize = 20
        ax = plt.subplot(1, 2, 1)
        _, cbar = utils.visualize(
            ax,
            coord[0, 0, :, :],
            coord[0, 1, :, :],
            output_v[0, :, :],
            "horizontal",
            [0, max(PARALIST)],
        )
        ax.set_aspect("equal")
        utils.set_axis_label(ax, "p")
        ax.set_title("PhyGeoNet " + r"$T$", fontsize=titlesize)
        ax.set_xlabel(xlabel=r"$x$", fontsize=xylabelsize)
        ax.set_ylabel(ylabel=r"$y$", fontsize=xylabelsize)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis="x", labelsize=xytickssize)
        ax.tick_params(axis="y", labelsize=xytickssize)
        cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
        cbar.ax.tick_params(labelsize=xytickssize)
        ax = plt.subplot(1, 2, 2)
        _, cbar = utils.visualize(
            ax,
            coord[0, 0, :, :],
            coord[0, 1, :, :],
            truth[0, 0, :, :],
            "horizontal",
            [0, max(PARALIST)],
        )
        ax.set_aspect("equal")
        utils.set_axis_label(ax, "p")
        ax.set_title("FV " + r"$T$", fontsize=titlesize)
        ax.set_xlabel(xlabel=r"$x$", fontsize=xylabelsize)
        ax.set_ylabel(ylabel=r"$y$", fontsize=xylabelsize)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis="x", labelsize=xytickssize)
        ax.tick_params(axis="y", labelsize=xytickssize)
        cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
        cbar.ax.tick_params(labelsize=xytickssize)
        fig1.tight_layout(pad=1)
        fig1.savefig(osp.join(cfg.output_dir, f"Para{i}T.png"), bbox_inches="tight")
        plt.close(fig1)


@hydra.main(
    version_base=None, config_path="./conf", config_name="heat_equation_with_bc.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

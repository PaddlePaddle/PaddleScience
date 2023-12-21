from os import path as osp

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def setAxisLabel(ax, type):
    if type == "p":
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    elif type == "r":
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
    else:
        raise ValueError("The axis type only can be reference or physical")


def gen_e2vcg(x):
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    nnx = x.shape[1]
    e2vcg = np.zeros([4, nelem])
    for j in range(nelemy):
        for i in range(nelemx):
            e2vcg[:, j * nelemx + i] = np.asarray(
                [j * nnx + i, j * nnx + i + 1, (j + 1) * nnx + i, (j + 1) * nnx + i + 1]
            )
    return e2vcg.astype("int")


def visualize2D(ax, x, y, u, colorbarPosition="vertical", colorlimit=None):
    xdg0 = np.vstack([x.flatten(order="C"), y.flatten(order="C")])
    udg0 = u.flatten(order="C")
    idx = np.asarray([0, 1, 3, 2])
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    e2vcg0 = gen_e2vcg(x)
    udg_ref = udg0[e2vcg0]
    cmap = matplotlib.cm.coolwarm
    polygon_list = []
    for i in range(nelem):
        polygon_ = Polygon(xdg0[:, e2vcg0[idx, i]].T)
        polygon_list.append(polygon_)
    polygon_ensemble = PatchCollection(polygon_list, cmap=cmap, alpha=1)
    polygon_ensemble.set_edgecolor("face")
    polygon_ensemble.set_array(np.mean(udg_ref, axis=0))
    if colorlimit is None:
        pass
    else:
        polygon_ensemble.set_clim(colorlimit)
    ax.add_collection(polygon_ensemble)
    ax.set_xlim(np.min(xdg0[0, :]), np.max(xdg0[0, :]))
    ax.set_ylim(np.min(xdg0[1, :]), np.max(xdg0[1, :]))
    cbar = plt.colorbar(polygon_ensemble, orientation=colorbarPosition)
    return ax, cbar


def dfdx(f, dydeta, dydxi, Jinv, h=0.01):
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )
    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdx = Jinv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(f, dxdxi, dxdeta, Jinv, h=0.01):
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )
    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdy = Jinv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def train(cfg: DictConfig):
    # initiallizer
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")
    pad_singleside = cfg.MODEL.pad_singleside

    model = ppsci.arch.USCNN(**cfg.MODEL)

    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.lr)(model)

    data = np.load(cfg.data_dir)
    coords = data["coords"]
    jinvs = data["jinvs"]
    dxdxis = data["dxdxis"]
    dydxis = data["dydxis"]
    dxdetas = data["dxdetas"]
    dydetas = data["dydetas"]
    len_data = cfg.len_data

    sup_constraint_mres = ppsci.constraint.SupervisedConstraint(
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
            "batch_size": 1,
            "iters_per_epoch": coords.shape[0],
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(lambda out, label, weught: out["mRes"]),
        name="mRes",
    )

    sup_constraint = {sup_constraint_mres.name: sup_constraint_mres}

    def _transform_out(_input, outputV, pad_singleside=pad_singleside, k=len_data):
        outputV = outputV["outputV"]
        batchSize = outputV.shape[0]
        Jinv = _input["jinvs"]
        dxdxi = _input["dxdxis"]
        dydxi = _input["dydxis"]
        dxdeta = _input["dxdetas"]
        dydeta = _input["dydetas"]
        Para = _input["coords"]
        for j in range(batchSize):
            outputV[j, 0, -pad_singleside:, pad_singleside:-pad_singleside] = outputV[
                j, 0, 1:2, pad_singleside:-pad_singleside
            ]
            outputV[j, 0, :pad_singleside, pad_singleside:-pad_singleside] = outputV[
                j, 0, -2:-1, pad_singleside:-pad_singleside
            ]
            outputV[j, 0, :, -pad_singleside:] = 0
            outputV[j, 0, :, 0:pad_singleside] = Para[j, 0, 0, 0]
        dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
        continuity = d2vdy2 + d2vdx2
        return {"mRes": paddle.mean(continuity**2) / k}

    model.register_output_transform(_transform_out)
    solver = ppsci.solver.Solver(
        model,
        sup_constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.epochs,
        iters_per_epoch=coords.shape[0],
    )

    solver.train()
    solver.plot_loss_history()


def evaluate(cfg: DictConfig):
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)

    pad_singleside = cfg.MODEL.pad_singleside

    model = ppsci.arch.USCNN(**cfg.MODEL)

    model.register_output_transform(None)
    data = np.load(cfg.test_data_dir)
    paras = paddle.to_tensor(data["paras"])
    truths = paddle.to_tensor(data["truths"])
    coords = paddle.to_tensor(data["coords"])
    len_data = cfg.len_data
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,  ### the path of the model
    )

    outputV = solver.predict({"coords": paras})
    outputV = outputV["outputV"]
    batchSize = outputV.shape[0]
    for j in range(batchSize):
        # Impose BC
        outputV[j, 0, -pad_singleside:, pad_singleside:-pad_singleside] = outputV[
            j, 0, 1:2, pad_singleside:-pad_singleside
        ]
        outputV[j, 0, :pad_singleside, pad_singleside:-pad_singleside] = outputV[
            j, 0, -2:-1, pad_singleside:-pad_singleside
        ]
        outputV[j, 0, :, -pad_singleside:] = 0
        outputV[j, 0, :, 0:pad_singleside] = paras[j, 0, 0, 0]
    eV = paddle.sqrt(
        paddle.mean((truths - outputV) ** 2) / paddle.mean(truths**2)
    ).item()
    print(eV / len_data)
    outputVs = outputV.numpy()
    ParaList = [1, 2, 3, 4, 5, 6, 7]
    for i in range(len(ParaList)):
        truth = truths[i].numpy()
        coord = coords[i].numpy()
        outputV = outputVs[i]
        truth = truth.reshape(1, 1, truth.shape[0], truth.shape[1])
        coord = coord.reshape(1, 2, coord.shape[2], coord.shape[3])
        print("i=", str(i))
        fig1 = plt.figure()
        xylabelsize = 20
        xytickssize = 20
        titlesize = 20
        ax = plt.subplot(1, 2, 1)
        _, cbar = visualize2D(
            ax,
            coord[0, 0, :, :],
            coord[0, 1, :, :],
            outputV[0, :, :],
            "horizontal",
            [0, max(ParaList)],
        )
        ax.set_aspect("equal")
        setAxisLabel(ax, "p")
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
        _, cbar = visualize2D(
            ax,
            coord[0, 0, :, :],
            coord[0, 1, :, :],
            truth[0, 0, :, :],
            "horizontal",
            [0, max(ParaList)],
        )
        ax.set_aspect("equal")
        setAxisLabel(ax, "p")
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
        fig1.savefig("Para" + str(i) + "T.png", bbox_inches="tight")
        plt.close(fig1)


@hydra.main(version_base=None, config_path="./conf", config_name="conf.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

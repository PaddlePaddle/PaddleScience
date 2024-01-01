from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from utils import dfdx
from utils import dfdy
from utils import setAxisLabel
from utils import visualize2D

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # initiallizer
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")
    data = np.load(cfg.data_dir)
    coords = data["coords"]
    jinvs = data["jinvs"]
    dxdxis = data["dxdxis"]
    dydxis = data["dydxis"]
    dxdetas = data["dxdetas"]
    dydetas = data["dydetas"]

    pad_singleside = cfg.MODEL.pad_singleside
    model = ppsci.arch.USCNN(**cfg.MODEL)

    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

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
            "batch_size": cfg.TRAIN.batch_size,
            "iters_per_epoch": coords.shape[0],
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(lambda out, label, weught: out["mRes"]),
        name="mRes",
    )

    sup_constraint = {sup_constraint_mres.name: sup_constraint_mres}

    def _transform_out(_input, _output, pad_singleside=pad_singleside):
        outputV = _output["outputV"]
        batchSize = outputV.shape[0]
        Jinv = _input["jinvs"]
        dxdxi = _input["dxdxis"]
        dydxi = _input["dydxis"]
        dxdeta = _input["dxdetas"]
        dydeta = _input["dydetas"]
        for j in range(batchSize):
            outputV[j, 0, -pad_singleside:, pad_singleside:-pad_singleside] = 0
            outputV[j, 0, :pad_singleside, pad_singleside:-pad_singleside] = 1
            outputV[j, 0, pad_singleside:-pad_singleside, -pad_singleside:] = 1
            outputV[j, 0, pad_singleside:-pad_singleside, 0:pad_singleside] = 1
            outputV[j, 0, 0, 0] = 0.5 * (outputV[j, 0, 0, 1] + outputV[j, 0, 1, 0])
            outputV[j, 0, 0, -1] = 0.5 * (outputV[j, 0, 0, -2] + outputV[j, 0, 1, -1])
        dvdx = dfdx(outputV, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(outputV, dxdxi, dxdeta, Jinv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, Jinv)
        continuity = d2vdy2 + d2vdx2
        return {"mRes": paddle.mean(continuity**2)}

    model.register_output_transform(_transform_out)
    output_dir = cfg.output_dir
    solver = ppsci.solver.Solver(
        model,
        sup_constraint,
        output_dir,
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

    data = np.load(cfg.data_dir)
    coords = data["coords"]

    OFV_sb = paddle.to_tensor(data["OFV_sb"])

    ## create model
    pad_singleside = cfg.MODEL.pad_singleside
    model = ppsci.arch.USCNN(**cfg.MODEL)
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,  ### the path of the model
    )
    outputV = solver.predict({"coords": paddle.to_tensor(coords)})
    outputV = outputV["outputV"]
    outputV[0, 0, -pad_singleside:, pad_singleside:-pad_singleside] = 0
    outputV[0, 0, :pad_singleside, pad_singleside:-pad_singleside] = 1
    outputV[0, 0, pad_singleside:-pad_singleside, -pad_singleside:] = 1
    outputV[0, 0, pad_singleside:-pad_singleside, 0:pad_singleside] = 1
    outputV[0, 0, 0, 0] = 0.5 * (outputV[0, 0, 0, 1] + outputV[0, 0, 1, 0])
    outputV[0, 0, 0, -1] = 0.5 * (outputV[0, 0, 0, -2] + outputV[0, 0, 1, -1])
    CNNVNumpy = outputV[0, 0, :, :]
    ev = paddle.sqrt(
        paddle.mean((OFV_sb - CNNVNumpy) ** 2) / paddle.mean(OFV_sb**2)
    ).item()
    logger.info(ev)
    outputV = outputV.numpy()
    OFV_sb = OFV_sb.numpy()
    fig1 = plt.figure()
    ax = plt.subplot(1, 2, 1)
    visualize2D(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        outputV[0, 0, 1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    setAxisLabel(ax, "p")
    ax.set_title("CNN " + r"$T$")
    ax.set_aspect("equal")
    ax = plt.subplot(1, 2, 2)
    visualize2D(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        OFV_sb[1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    setAxisLabel(ax, "p")
    ax.set_aspect("equal")
    ax.set_title("FV " + r"$T$")
    fig1.tight_layout(pad=1)
    fig1.savefig("T.png", bbox_inches="tight")
    plt.close(fig1)


@hydra.main(version_base=None, config_path="./conf", config_name="heat_equation.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

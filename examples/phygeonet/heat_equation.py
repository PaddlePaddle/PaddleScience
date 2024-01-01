from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig
from utils import dfdx
from utils import dfdy
from utils import setAxisLabel
from utils import visualize

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

    iters_per_epoch = coords.shape[0]
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
            "iters_per_epoch": iters_per_epoch,
            "num_workers": 0,
        },
        ppsci.loss.FunctionalLoss(lambda out, label, weught: out["mRes"]),
        name="mRes",
    )
    sup_constraint = {sup_constraint_mres.name: sup_constraint_mres}

    def _transform_out(_input, _output, pad_singleside=pad_singleside):
        output_v = _output["outputV"]
        Jinv = _input["jinvs"]
        dxdxi = _input["dxdxis"]
        dydxi = _input["dydxis"]
        dxdeta = _input["dxdetas"]
        dydeta = _input["dydetas"]
        output_v[:, 0, -pad_singleside:, pad_singleside:-pad_singleside] = 0
        output_v[:, 0, :pad_singleside, pad_singleside:-pad_singleside] = 1
        output_v[:, 0, pad_singleside:-pad_singleside, -pad_singleside:] = 1
        output_v[:, 0, pad_singleside:-pad_singleside, 0:pad_singleside] = 1
        output_v[:, 0, 0, 0] = 0.5 * (output_v[:, 0, 0, 1] + output_v[:, 0, 1, 0])
        output_v[:, 0, 0, -1] = 0.5 * (output_v[:, 0, 0, -2] + output_v[:, 0, 1, -1])
        dvdx = dfdx(output_v, dydeta, dydxi, Jinv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, Jinv)
        dvdy = dfdy(output_v, dxdxi, dxdeta, Jinv)
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
        iters_per_epoch=iters_per_epoch,
        seed=SEED,
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
    output_v = solver.predict({"coords": paddle.to_tensor(coords)})
    output_v = output_v["outputV"]

    output_v[0, 0, -pad_singleside:, pad_singleside:-pad_singleside] = 0
    output_v[0, 0, :pad_singleside, pad_singleside:-pad_singleside] = 1
    output_v[0, 0, pad_singleside:-pad_singleside, -pad_singleside:] = 1
    output_v[0, 0, pad_singleside:-pad_singleside, 0:pad_singleside] = 1
    output_v[0, 0, 0, 0] = 0.5 * (output_v[0, 0, 0, 1] + output_v[0, 0, 1, 0])
    output_v[0, 0, 0, -1] = 0.5 * (output_v[0, 0, 0, -2] + output_v[0, 0, 1, -1])

    ev = paddle.sqrt(
        paddle.mean((OFV_sb - output_v[0, 0]) ** 2) / paddle.mean(OFV_sb**2)
    ).item()
    logger.info(f"ev: {ev}")

    output_v = output_v.numpy()
    OFV_sb = OFV_sb.numpy()
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    visualize(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        output_v[0, 0, 1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    setAxisLabel(ax, "p")
    ax.set_title("CNN " + r"$T$")
    ax.set_aspect("equal")
    ax = plt.subplot(1, 2, 2)
    visualize(
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
    fig.tight_layout(pad=1)
    fig.savefig(f"{cfg.output_dir}/result.png", bbox_inches="tight")
    plt.close(fig)


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

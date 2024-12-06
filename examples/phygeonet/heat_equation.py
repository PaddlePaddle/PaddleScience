import os.path as osp
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
    data = np.load(cfg.data_dir)
    coords = data["coords"]
    jinvs = data["jinvs"]
    dxdxis = data["dxdxis"]
    dydxis = data["dydxis"]
    dxdetas = data["dxdetas"]
    dydetas = data["dydetas"]

    model = ppsci.arch.USCNN(**cfg.MODEL)

    optimizer = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model)

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
        jinv = _input["jinvs"]
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
        cfg.output_dir,
        optimizer,
        epochs=cfg.epochs,
        iters_per_epoch=iters_per_epoch,
    )
    solver.train()
    solver.plot_loss_history()


def evaluate(cfg: DictConfig):
    data = np.load(cfg.data_dir)
    coords = data["coords"]

    ofv_sb = paddle.to_tensor(data["OFV_sb"])

    ## create model
    pad_singleside = cfg.MODEL.pad_singleside
    model = ppsci.arch.USCNN(**cfg.MODEL)
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,  ### the path of the model
    )
    output_v = solver.predict({"coords": paddle.to_tensor(coords)})
    output_v = output_v["output_v"]

    output_v[0, 0, -pad_singleside:, pad_singleside:-pad_singleside] = 0
    output_v[0, 0, :pad_singleside, pad_singleside:-pad_singleside] = 1
    output_v[0, 0, pad_singleside:-pad_singleside, -pad_singleside:] = 1
    output_v[0, 0, pad_singleside:-pad_singleside, 0:pad_singleside] = 1
    output_v[0, 0, 0, 0] = 0.5 * (output_v[0, 0, 0, 1] + output_v[0, 0, 1, 0])
    output_v[0, 0, 0, -1] = 0.5 * (output_v[0, 0, 0, -2] + output_v[0, 0, 1, -1])

    ev = paddle.sqrt(
        paddle.mean((ofv_sb - output_v[0, 0]) ** 2) / paddle.mean(ofv_sb**2)
    ).item()
    logger.info(f"ev: {ev}")

    output_v = output_v.numpy()
    ofv_sb = ofv_sb.numpy()
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    utils.visualize(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        output_v[0, 0, 1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    utils.set_axis_label(ax, "p")
    ax.set_title("CNN " + r"$T$")
    ax.set_aspect("equal")
    ax = plt.subplot(1, 2, 2)
    utils.visualize(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        ofv_sb[1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    utils.set_axis_label(ax, "p")
    ax.set_aspect("equal")
    ax.set_title("FV " + r"$T$")
    fig.tight_layout(pad=1)
    fig.savefig(f"{cfg.output_dir}/result.png", bbox_inches="tight")
    plt.close(fig)


def export(cfg: DictConfig):
    model = ppsci.arch.USCNN(**cfg.MODEL)
    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None, 2, 19, 84], "float32", name=key)
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)
    data = np.load(cfg.data_dir)
    coords = data["coords"]
    ofv_sb = data["OFV_sb"]

    ## create model
    pad_singleside = cfg.MODEL.pad_singleside
    input_spec = {"coords": coords}

    output_v = predictor.predict(input_spec, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_v = {
        store_key: output_v[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_v.keys())
    }

    output_v = output_v["output_v"]

    output_v[0, 0, -pad_singleside:, pad_singleside:-pad_singleside] = 0
    output_v[0, 0, :pad_singleside, pad_singleside:-pad_singleside] = 1
    output_v[0, 0, pad_singleside:-pad_singleside, -pad_singleside:] = 1
    output_v[0, 0, pad_singleside:-pad_singleside, 0:pad_singleside] = 1
    output_v[0, 0, 0, 0] = 0.5 * (output_v[0, 0, 0, 1] + output_v[0, 0, 1, 0])
    output_v[0, 0, 0, -1] = 0.5 * (output_v[0, 0, 0, -2] + output_v[0, 0, 1, -1])

    ev = paddle.sqrt(
        paddle.mean((ofv_sb - output_v[0, 0]) ** 2) / paddle.mean(ofv_sb**2)
    ).item()
    logger.info(f"ev: {ev}")

    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    utils.visualize(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        output_v[0, 0, 1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    utils.set_axis_label(ax, "p")
    ax.set_title("CNN " + r"$T$")
    ax.set_aspect("equal")
    ax = plt.subplot(1, 2, 2)
    utils.visualize(
        ax,
        coords[0, 0, 1:-1, 1:-1],
        coords[0, 1, 1:-1, 1:-1],
        ofv_sb[1:-1, 1:-1],
        "horizontal",
        [0, 1],
    )
    utils.set_axis_label(ax, "p")
    ax.set_aspect("equal")
    ax.set_title("FV " + r"$T$")
    fig.tight_layout(pad=1)
    fig.savefig(osp.join(cfg.output_dir, "result.png"), bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="./conf", config_name="heat_equation.yaml")
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

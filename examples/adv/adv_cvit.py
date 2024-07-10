"""
Reference: https://github.com/PredictiveIntelligenceLab/cvit/tree/main/adv/
"""

from os import path as osp

import einops
import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger

dtype = paddle.get_default_dtype()


def plot_result(pred: np.ndarray, label: np.ndarray, output_dir: str):
    def compute_tvd(f, g, dx):
        assert f.shape == g.shape
        df = np.abs(np.diff(f, axis=1))
        dg = np.abs(np.diff(g, axis=1))

        tvd = np.sum(np.abs(df - dg), axis=1) * dx
        return tvd

    tvd = compute_tvd(np.squeeze(pred, axis=-1), label, 1 / 199)
    logger.message(
        f"mean: {np.mean(tvd)}, "
        f"median: {np.median(tvd)}, "
        f"max: {np.amax(tvd)}, "
        f"min: {np.amin(tvd)}"
    )

    best_idx = np.argmin(tvd)
    worst_idx = np.argmax(tvd)
    logger.message(f"best: {best_idx}, worst: {worst_idx}")

    idx = worst_idx
    x = np.linspace(0, 1, 200)
    plt.plot(x, pred[idx], "r--")
    plt.plot(x, label[idx], "b-")
    plt.title(f"CViT (TV: {tvd[idx]:.2f})")
    plt.xlabel("$y$")
    plt.ylim([-1.4, 1.4])

    plt.tight_layout()
    plt.savefig(osp.join(output_dir, "adv_cvit.png"))
    logger.message(f"Result saved to: {osp.join(output_dir, 'adv_cvit.png')}")


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.CVit1D(**cfg.MODEL)

    # prepare dataset
    inputs = np.load(osp.join(cfg.DATA_DIR, "adv_a0.npy")).astype(dtype)
    outputs = np.load(osp.join(cfg.DATA_DIR, "adv_aT.npy")).astype(dtype)
    grid = np.linspace(0, 1, inputs.shape[0], dtype=dtype)
    grid = einops.repeat(grid, "i -> i b", b=inputs.shape[1])

    ## swapping the first two axes:
    inputs = einops.rearrange(inputs, "i j -> j i 1")  # (40000, 200, 1)
    outputs = einops.rearrange(outputs, "i j -> j i")  # (40000, 200)
    grid = einops.rearrange(grid, "i j -> j i 1")  # (40000, 200, 1)

    idx = np.random.permutation(inputs.shape[0])
    n_train = 20000
    n_test = 10000
    inputs_train, outputs_train, grid_train = (
        inputs[idx[:n_train]],
        outputs[idx[:n_train]],
        grid[idx[:n_train]],
    )
    inputs_test, outputs_test, grid_test = (
        inputs[idx[-n_test:]],
        outputs[idx[-n_test:]],
        grid[idx[-n_test:]],
    )

    # set constraint
    def gen_input_batch_train():
        batch_idx = np.random.randint(0, inputs_train.shape[0], [cfg.TRAIN.batch_size])
        grid_idx = np.sort(
            np.random.randint(0, inputs_train.shape[1], [cfg.TRAIN.grid_size])
        )
        return {
            "u": inputs_train[batch_idx],
            "y": grid_train[batch_idx][:, grid_idx],
            "batch_idx": batch_idx,
            "grid_idx": grid_idx,
        }

    def gen_label_batch_train(input_batch):
        batch_idx, grid_idx = input_batch.pop("batch_idx"), input_batch.pop("grid_idx")
        return {
            "s": outputs_train[batch_idx][:, grid_idx, None],
        }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "ContinuousNamedArrayDataset",
                "input": gen_input_batch_train,
                "label": gen_label_batch_train,
            },
        },
        output_expr={"s": lambda out: out["s"]},
        loss=ppsci.loss.MSELoss("mean"),
        name="Sup",
    )
    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        **cfg.TRAIN.lr_scheduler
    )()
    optimizer = ppsci.optimizer.AdamW(
        lr_scheduler,
        weight_decay=cfg.TRAIN.weight_decay,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(cfg.TRAIN.grad_clip),
    )(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        cfg=cfg,
    )
    # train model
    solver.train()
    # visualzie result on ema model
    solver.ema_model.apply_shadow()
    pred_s = solver.predict(
        {"u": inputs_test, "y": grid_test},
        batch_size=cfg.EVAL.batch_size,
        return_numpy=True,
    )["s"]

    plot_result(pred_s, outputs_test, cfg.output_dir)
    solver.ema_model.restore()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.CVit1D(**cfg.MODEL)

    # prepare dataset
    inputs = np.load(osp.join(cfg.DATA_DIR, "adv_a0.npy")).astype(dtype)
    outputs = np.load(osp.join(cfg.DATA_DIR, "adv_aT.npy")).astype(dtype)
    grid = np.linspace(0, 1, inputs.shape[0], dtype=dtype)
    grid = einops.repeat(grid, "i -> i b", b=inputs.shape[1])

    ## swapping the first two axes:
    inputs = einops.rearrange(inputs, "i j -> j i 1")  # (40000, 200, 1)
    outputs = einops.rearrange(outputs, "i j -> j i")  # (40000, 200)
    grid = einops.rearrange(grid, "i j -> j i 1")  # (40000, 200, 1)

    idx = np.random.permutation(inputs.shape[0])
    n_test = 10000
    inputs_test, outputs_test, grid_test = (
        inputs[idx[-n_test:]],
        outputs[idx[-n_test:]],
        grid[idx[-n_test:]],
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
    )
    pred_s = solver.predict(
        {"u": inputs_test, "y": grid_test},
        batch_size=cfg.EVAL.batch_size,
        return_numpy=True,
    )["s"]

    plot_result(pred_s, outputs_test, cfg.output_dir)


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.CVit1D(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(model, cfg=cfg)
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            model.input_keys[0]: InputSpec(
                [None, cfg.INFER.spatial_dims, 1],
                name=model.input_keys[0],
            ),
            model.input_keys[1]: InputSpec(
                [None, cfg.INFER.grid_size[0], 1],
                name=model.input_keys[1],
            ),
        },
    ]
    # NOTE: Put einops into ignore module when exporting, or error will occur
    solver.export(
        input_spec, cfg.INFER.export_path, with_onnx=False, ignore_modules=[einops]
    )


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # prepare dataset
    inputs = np.load(osp.join(cfg.DATA_DIR, "adv_a0.npy")).astype(dtype)
    outputs = np.load(osp.join(cfg.DATA_DIR, "adv_aT.npy")).astype(dtype)
    grid = np.linspace(0, 1, inputs.shape[0], dtype=dtype)
    grid = einops.repeat(grid, "i -> i b", b=inputs.shape[1])

    ## swapping the first two axes:
    inputs = einops.rearrange(inputs, "i j -> j i 1")  # (40000, 200, 1)
    outputs = einops.rearrange(outputs, "i j -> j i")  # (40000, 200)
    grid = einops.rearrange(grid, "i j -> j i 1")  # (40000, 200, 1)

    idx = np.random.permutation(inputs.shape[0])
    n_test = 10000
    inputs_test, outputs_test, grid_test = (
        inputs[idx[-n_test:]],
        outputs[idx[-n_test:]],
        grid[idx[-n_test:]],
    )

    output_dict = predictor.predict(
        {"u": inputs_test, "y": grid_test},
        batch_size=cfg.INFER.batch_size,
    )
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    plot_result(output_dict[cfg.MODEL.output_keys[0]], outputs_test, "./")


@hydra.main(version_base=None, config_path="./conf", config_name="adv_cvit.yaml")
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

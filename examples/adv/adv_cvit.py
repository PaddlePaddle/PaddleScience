"""
Reference: https://github.com/PredictiveIntelligenceLab/cvit/tree/main/adv/
"""

from os import path as osp

import einops
import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci

dtype = paddle.get_default_dtype()


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.CVit1D(**cfg.MODEL)

    # prepare dataset
    inputs = np.load(osp.join(cfg.DATA_DIR, "adv_a0.npy")).astype(np.float32)
    outputs = np.load(osp.join(cfg.DATA_DIR, "adv_aT.npy")).astype(np.float32)
    grid = np.linspace(0, 1, inputs.shape[0], dtype=np.float32)
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

    def gen_input_batch_train():
        batch_idx = np.random.randint(0, inputs_train.shape[0], [cfg.TRAIN.batch_size])
        grid_idx = np.sort(
            np.random.randint(0, inputs_train.shape[1], [cfg.TRAIN.grid_size])
        )
        return {
            "u": inputs_train[batch_idx],  # [N, 200, 1]
            "y": grid_train[batch_idx][:, grid_idx],  # [N, G, 1]
            "batch_idx": batch_idx,
            "grid_idx": grid_idx,
        }

    def gen_label_batch_train(input_batch):
        batch_idx, grid_idx = input_batch.pop("batch_idx"), input_batch.pop("grid_idx")
        return {
            "s": outputs_train[batch_idx][:, grid_idx, None],  # [N, G, 1]
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

    # set validator
    def avg_l2_metric_func(output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            # reshape to [B, L]
            x, y = output_dict[key].squeeze(-1), label_dict[key].squeeze(-1)
            # compute metrics along all samples
            l2_err = (x - y).norm(p=2, axis=-1) / y.norm(p=2, axis=-1)
            metric_dict[f"{key}.min"] = l2_err.min()
            metric_dict[f"{key}.median"] = l2_err.median()
            metric_dict[f"{key}.mean"] = l2_err.mean()
            metric_dict[f"{key}.max"] = l2_err.max()
        return metric_dict

    u_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"u": inputs_test, "y": grid_test},
                "label": {"s": outputs_test[..., None]},
            },
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"s": lambda out: out["s"]},
        metric={"L2Rel": ppsci.metric.FunctionalMetric(avg_l2_metric_func)},
        name="s_validator",
    )
    validator = {u_validator.name: u_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )
    solver.eval()
    # train model
    solver.train()
    # # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.CVit1D(**cfg.MODEL)

    # prepare dataset
    inputs = np.load(osp.join(cfg.DATA_DIR, "adv_a0.npy")).astype(np.float32)
    outputs = np.load(osp.join(cfg.DATA_DIR, "adv_aT.npy")).astype(np.float32)
    grid = np.linspace(0, 1, inputs.shape[0], dtype=np.float32)
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

    # set validator
    def avg_l2_metric_func(output_dict, label_dict):
        metric_dict = {}
        for key in label_dict:
            # reshape to [B, L]
            x, y = output_dict[key].squeeze(-1), label_dict[key].squeeze(-1)
            # compute metrics along all samples
            l2_err = (x - y).norm(p=2, axis=-1) / y.norm(p=2, axis=-1)
            metric_dict[f"{key}.mean"] = l2_err.mean()
            metric_dict[f"{key}.median"] = l2_err.median()
            metric_dict[f"{key}.min"] = l2_err.min()
            metric_dict[f"{key}.max"] = l2_err.max()
        return metric_dict

    u_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"u": inputs_test, "y": grid_test},
                "label": {"s": outputs_test[..., None]},
            },
            "batch_size": cfg.EVAL.batch_size,
        },
        ppsci.loss.MSELoss("mean"),
        {"s": lambda out: out["s"]},
        metric={"L2Rel": ppsci.metric.FunctionalMetric(avg_l2_metric_func)},
        name="s_validator",
    )
    validator = {u_validator.name: u_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    solver.eval()


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
                [None, cfg.INFER.seq_len, 1],
                name=model.input_keys[0],
            ),
            model.input_keys[1]: InputSpec(
                [None, cfg.INFER.grid_size, 1],
                name=model.input_keys[1],
            ),
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)


def inference(cfg: DictConfig):
    pass


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

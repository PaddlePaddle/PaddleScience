"""
Reference: https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main/examples/allen_cahn
"""

from os import path as osp

import einops
import hydra
import numpy as np
import paddle
import scipy.io as sio
from omegaconf import DictConfig

import ppsci
from ppsci.utils import misc

dtype = paddle.get_default_dtype()


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.CVit1D(**cfg.MODEL)
    # for k, v in model.named_parameters():
    #     print("param", k, v.shape, v.numel().item())
    # for k, v in model.named_buffers():
    #     print("buffer", k, v.shape, v.numel().item())
    print(model)
    # exit()
    # prepare dataset
    inputs = np.load(osp.join(cfg.DATA_DIR, "adv_a0.npy")).astype(np.float32)
    outputs = np.load(osp.join(cfg.DATA_DIR, "adv_aT.npy")).astype(np.float32)
    grid = np.linspace(0, 1, inputs.shape[0], dtype=np.float32)
    grid = einops.repeat(grid, "i -> i b", b=inputs.shape[1])

    ## swapping the first two axes:
    inputs = einops.rearrange(inputs, "i j -> j i 1")  # (40000, 200, 1)
    grid = einops.rearrange(grid, "i j -> j i 1")  # (40000, 200, 1)
    outputs = einops.rearrange(outputs, "i j -> j i")  # (40000, 200)

    ## split the data into training, validation, and test sets
    idx = np.random.permutation(inputs.shape[0])
    n_train = 20000
    n_val = 10000
    # n_test = 10000
    inputs_train, outputs_train, grid_train = (
        inputs[idx[:n_train]],
        outputs[idx[:n_train]],
        grid[idx[:n_train]],
    )
    inputs_val, outputs_val, grid_val = (
        inputs[idx[n_train : n_train + n_val]],
        outputs[idx[n_train : n_train + n_val]],
        grid[idx[n_train : n_train + n_val]],
    )
    # inputs_test, outputs_test, grid_test = (
    #     inputs[idx[-n_test:]],
    #     outputs[idx[-n_test:]],
    #     grid[idx[-n_test:]],
    # )

    def gen_input_batch_train():
        batch_idx = np.random.randint(0, inputs_train.shape[0], [cfg.TRAIN.batch_size])
        grid_idx = np.sort(
            np.random.randint(0, inputs_train.shape[1], [cfg.TRAIN.grid_size])
        )
        return {
            "u": inputs_train[batch_idx, :],  # [N, 200, 1]
            "y": grid_train[batch_idx, :][:, grid_idx],  # [N, G, 1]
            "batch_idx": batch_idx,
            "grid_idx": grid_idx,
        }

    def gen_label_batch_train(input_batch):
        batch_idx, grid_idx = input_batch.pop("batch_idx"), input_batch.pop("grid_idx")
        return {
            "s": outputs_train[batch_idx, :][:, grid_idx, None],  # [N, G]
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
        weight_decay=1e-5,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
    )(model)

    # set validator
    def gen_input_batch_val():
        batch_idx = np.random.randint(0, inputs_val.shape[0], [cfg.EVAL.batch_size])
        grid_idx = np.sort(
            np.random.randint(0, inputs_val.shape[1], [cfg.EVAL.grid_size])
        )
        return {
            "u": inputs_val[batch_idx, :],  # [N, 200, 1]
            "y": grid_val[batch_idx, :][:, grid_idx],  # [N, G, 1]
            "batch_idx": batch_idx,
            "grid_idx": grid_idx,
        }

    def gen_label_batch_val(input_batch):
        batch_idx, grid_idx = input_batch["batch_idx"], input_batch["grid_idx"]
        return {
            "s": outputs_val[batch_idx, :][:, grid_idx, None],  # [N, G]
        }

    inputs_val_all = []
    label_val_all = []
    for _ in range(n_val // cfg.EVAL.batch_size):
        input_batch = gen_input_batch_val()
        label_batch = gen_label_batch_val(input_batch)
        input_batch.pop("batch_idx")
        input_batch.pop("grid_idx")
        inputs_val_all.append(input_batch)
        label_val_all.append(label_batch)
    inputs_val_all = ppsci.utils.misc.concat_dict_list(inputs_val_all)
    label_val_all = ppsci.utils.misc.concat_dict_list(label_val_all)

    def avg_l2_metric_func(output_dict, label_dict):
        error = 0.0
        for key in label_dict:
            for bid in range(len(label_dict[key]) // cfg.EVAL.batch_size):
                st, ed = bid * cfg.EVAL.batch_size, (bid + 1) * cfg.EVAL.batch_size
                error += ppsci.metric.L2Rel.l2_relative_error(
                    output_dict[key][st:ed, :, 0],
                    label_dict[key][st:ed, :, 0],
                )
        error /= len(label_dict[key]) // cfg.EVAL.batch_size
        return {"s": paddle.to_tensor(error)}

    u_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": inputs_val_all,
                "label": label_val_all,
            },
            "batch_size": 256,
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
    # evaluate after finished training
    solver.eval()


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

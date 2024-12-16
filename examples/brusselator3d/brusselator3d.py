"""
Paper: https://arxiv.org/abs/2303.10528
Reference: https://github.com/qianyingcao/Laplace-Neural-Operator/tree/main/3D_Brusselator
"""
from os import path as osp
from typing import List
from typing import Literal
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import reader


class DataFuncs:
    def __init__(self, orig_r: int, r: int, nt: int, nx: int, ny: int) -> None:
        """Functions of data.

        Args:
            orig_r (int): Oringinal resolution of data.
            r (int): Multiples of downsampling at resolution.
            nt (int): The number of values to take on t.
            nx (int): The number of values to take on x.
            ny (int): The number of values to take on y.
        """
        self.orig_r = orig_r
        self.r = r
        self.nt = nt
        self.nx = nx
        self.ny = ny

        self.s = int((orig_r - 1) / r + 1)

        x = np.linspace(0, 1, orig_r)
        y = np.linspace(0, 1, orig_r)
        t = np.linspace(0, 1, nt)
        self.tt, self.xx, self.yy = np.meshgrid(t, x, y, indexing="ij")

    def load_data(self, data_path, keys) -> List[np.ndarray]:
        raw_data = reader.load_npz_file(data_path, keys)
        return [raw_data[key] for key in keys]

    def get_mean_std(self, data: np.ndarray) -> Tuple[float, ...]:
        min_ = np.min(data)
        max_ = np.max(data)
        return (min_ + max_) / 2, (max_ - min_) / 2

    def encode(self, data, mean, std) -> np.ndarray:
        return (data - mean) / std

    def decode(self, data, mean, std) -> np.ndarray:
        return data * std + mean

    def gen_grid(self, grid, num) -> np.ndarray:
        grid_tile = np.tile(grid, (num, 1, 1, 1))
        grid_subsampling = grid_tile[:, :, :: self.r, :: self.r]
        grid_crop = grid_subsampling[:, :, : self.s, : self.s]
        grid_reshape = np.reshape(grid_crop, (num, self.nt, self.s, self.s, 1))
        return grid_reshape

    def cat_grid(self, data) -> np.ndarray:
        grid_t = self.gen_grid(self.tt, data.shape[0])
        grid_x = self.gen_grid(self.xx, data.shape[0])
        grid_y = self.gen_grid(self.yy, data.shape[0])
        return np.concatenate([data, grid_t, grid_x, grid_y], axis=-1).astype(
            data.dtype
        )

    def transform(
        self, data: np.ndarray, key: Literal["input", "label"] = "input"
    ) -> np.ndarray:
        if key == "input":
            data_expand = np.expand_dims(data, axis=0)
            data_tile = np.tile(data_expand, (self.orig_r, self.orig_r, 1, 1))
            data = np.transpose(data_tile, axes=(2, 3, 0, 1))
        data_subsampling = data[:, :, :: self.r, :: self.r]
        data_crop = data_subsampling[:, :, : self.s, : self.s]
        data_reshape = np.reshape(
            data_crop, (data.shape[0], self.nt, self.s, self.s, 1)
        )
        return data_reshape

    def draw_plot(self, save_path, pred, label):
        pred = np.mean(pred, axis=(1, 2))
        label = np.mean(label, axis=(1, 2))
        t = np.linspace(0, self.nt, self.nt)
        plt.figure(figsize=(8, 6))
        plt.plot(t, pred, label="pred(t)")
        plt.plot(t, label, label="label(t)")
        plt.xlabel("time steps")
        plt.legend()
        plt.savefig(save_path)


def train(cfg: DictConfig):
    # set data functions
    data_funcs = DataFuncs(cfg.ORIG_R, cfg.RESOLUTION, cfg.NUM_T, cfg.NUM_X, cfg.NUM_Y)
    inputs_train, labels_train, inputs_val, labels_val = data_funcs.load_data(
        cfg.DATA_PATH,
        ("inputs_train", "outputs_train", "inputs_test", "outputs_test"),
    )
    in_train = data_funcs.transform(inputs_train, "input")
    label_train = data_funcs.transform(labels_train, "label")
    in_val = data_funcs.transform(inputs_val, "input")
    label_val = data_funcs.transform(labels_val, "label")
    in_train_mean, in_train_std = data_funcs.get_mean_std(in_train)
    label_train_mean, label_train_std = data_funcs.get_mean_std(label_train)

    input_constraint = data_funcs.encode(in_train, in_train_mean, in_train_std)
    input_validator = data_funcs.encode(in_val, in_train_mean, in_train_std)
    if not cfg.MODEL.use_grid:
        input_constraint = data_funcs.cat_grid(input_constraint)
        input_validator = data_funcs.cat_grid(input_validator)

    # set model
    T = paddle.linspace(start=0, stop=19, num=cfg.NUM_T).reshape([1, cfg.NUM_T])
    X = paddle.linspace(start=0, stop=1, num=cfg.ORIG_R).reshape([1, cfg.ORIG_R])[
        :, : data_funcs.s
    ]
    Y = paddle.linspace(start=0, stop=1, num=cfg.ORIG_R).reshape([1, cfg.ORIG_R])[
        :, : data_funcs.s
    ]
    model = ppsci.arch.LNO(**cfg.MODEL, T=T, data=(X, Y))

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Step(**cfg.TRAIN.lr_scheduler)()
    optimizer = ppsci.optimizer.AdamW(
        lr_scheduler, weight_decay=cfg.TRAIN.weight_decay
    )(model)

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": input_constraint},
                "label": {
                    "output": data_funcs.encode(
                        label_train, label_train_mean, label_train_std
                    )
                },
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
            "num_workers": 1,
        },
        ppsci.loss.L2RelLoss("sum"),
        name="sup_constraint",
    )

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": input_validator},
                "label": {"output": label_val},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": 1,
        },
        ppsci.loss.L2RelLoss("sum"),
        {
            "output": lambda out: data_funcs.decode(
                out["output"],
                label_train_mean,
                label_train_std,
            )
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="sup_validator",
    )

    # wrap validator together
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        optimizer=optimizer,
        validator=validator,
        cfg=cfg,
    )

    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set data functions
    data_funcs = DataFuncs(cfg.ORIG_R, cfg.RESOLUTION, cfg.NUM_T, cfg.NUM_X, cfg.NUM_Y)
    inputs_train, labels_train, inputs_val, labels_val = data_funcs.load_data(
        cfg.DATA_PATH,
        ("inputs_train", "outputs_train", "inputs_test", "outputs_test"),
    )
    in_train = data_funcs.transform(inputs_train, "input")
    label_train = data_funcs.transform(labels_train, "label")
    in_val = data_funcs.transform(inputs_val, "input")
    label_val = data_funcs.transform(labels_val, "label")
    in_train_mean, in_train_std = data_funcs.get_mean_std(in_train)
    label_train_mean, label_train_std = data_funcs.get_mean_std(label_train)

    input_validator = data_funcs.encode(in_val, in_train_mean, in_train_std)
    if not cfg.MODEL.use_grid:
        input_validator = data_funcs.cat_grid(input_validator)

    # set model
    T = paddle.linspace(start=0, stop=19, num=cfg.NUM_T).reshape([1, cfg.NUM_T])
    X = paddle.linspace(start=0, stop=1, num=cfg.ORIG_R).reshape([1, cfg.ORIG_R])[
        :, : data_funcs.s
    ]
    Y = paddle.linspace(start=0, stop=1, num=cfg.ORIG_R).reshape([1, cfg.ORIG_R])[
        :, : data_funcs.s
    ]
    model = ppsci.arch.LNO(**cfg.MODEL, T=T, data=(X, Y))

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": input_validator},
                "label": {"output": label_val},
            },
            "batch_size": cfg.EVAL.batch_size,
            "num_workers": 1,
        },
        ppsci.loss.L2RelLoss("sum"),
        {
            "output": lambda out: data_funcs.decode(
                out["output"],
                label_train_mean,
                label_train_std,
            )
        },
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="sup_validator",
    )

    # wrap validator together
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        cfg=cfg,
    )
    # evaluate
    solver.eval()

    # visualize prediction
    input_visualize = data_funcs.encode(in_val[0:1], in_train_mean, in_train_std)
    if not cfg.MODEL.use_grid:
        input_visualize = data_funcs.cat_grid(input_visualize)
    output_dict = model({"input": paddle.to_tensor(input_visualize)})
    pred = paddle.squeeze(
        data_funcs.decode(output_dict["output"], label_train_mean, label_train_std)
    ).numpy()
    label = np.squeeze(label_val[0])

    data_funcs.draw_plot(osp.join(cfg.output_dir, "result"), pred, label)


def export(cfg: DictConfig):
    # set model
    T = paddle.linspace(start=0, stop=19, num=cfg.NUM_T).reshape([1, cfg.NUM_T])
    X = paddle.linspace(start=0, stop=1, num=cfg.ORIG_R).reshape([1, cfg.ORIG_R])[
        :, : int((cfg.ORIG_R - 1) / cfg.RESOLUTION + 1)
    ]
    Y = paddle.linspace(start=0, stop=1, num=cfg.ORIG_R).reshape([1, cfg.ORIG_R])[
        :, : int((cfg.ORIG_R - 1) / cfg.RESOLUTION + 1)
    ]
    model = ppsci.arch.LNO(**cfg.MODEL, T=T, data=(X, Y))

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )

    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec(
                [
                    None,
                    cfg.NUM_T,
                    cfg.NUM_X // cfg.RESOLUTION,
                    cfg.NUM_Y // cfg.RESOLUTION,
                    1,
                ],
                "float32",
                name=key,
            )
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # set data functions
    data_funcs = DataFuncs(cfg.ORIG_R, cfg.RESOLUTION, cfg.NUM_T, cfg.NUM_X, cfg.NUM_Y)
    inputs_train, labels_train, inputs_val, labels_val = data_funcs.load_data(
        cfg.DATA_PATH,
        ("inputs_train", "outputs_train", "inputs_test", "outputs_test"),
    )
    in_train = data_funcs.transform(inputs_train, "input")
    label_train = data_funcs.transform(labels_train, "label")
    in_val = data_funcs.transform(inputs_val, "input")
    label_val = data_funcs.transform(labels_val, "label")
    in_train_mean, in_train_std = data_funcs.get_mean_std(in_train)
    label_train_mean, label_train_std = data_funcs.get_mean_std(label_train)
    input_infer = data_funcs.encode(in_val[0:1], in_train_mean, in_train_std)
    if not cfg.MODEL.use_grid:
        input_infer = data_funcs.cat_grid(input_infer)

    output_dict = predictor.predict(
        {"input": input_infer},
        cfg.INFER.batch_size,
    )

    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.output_keys, output_dict.keys())
    }

    pred = np.squeeze(
        data_funcs.decode(output_dict["output"], label_train_mean, label_train_std)
    )
    label = np.squeeze(label_val[0])

    data_funcs.draw_plot(osp.join(cfg.output_dir, "result"), pred, label)


@hydra.main(version_base=None, config_path="./conf", config_name="brusselator3d.yaml")
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
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

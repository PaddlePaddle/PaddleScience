# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.loss import L2RelLoss
from ppsci.optimizer import Adam
from ppsci.optimizer import lr_scheduler
from ppsci.utils import logger


# build data
def getdata(
    x_path,
    y_path,
    para_path,
    output_path,
    n_data,
    n,
    s,
    is_train=True,
    is_inference=False,
):
    # load data
    inputX_raw = np.load(x_path)[:, 0:n_data]
    inputY_raw = np.load(y_path)[:, 0:n_data]
    inputPara_raw = np.load(para_path)[:, 0:n_data]
    output_raw = np.load(output_path)[:, 0:n_data]

    # preprocess data
    inputX = inputX_raw[:, 0::3]
    inputY = inputY_raw[:, 0::3]
    inputPara = inputPara_raw[:, 0::3]
    label = (output_raw[:, 0::3] + output_raw[:, 1::3] + output_raw[:, 2::3]) / 3.0

    if is_inference:
        inputX = np.transpose(inputX, (1, 0))
        inputY = np.transpose(inputY, (1, 0))
        input = np.stack(arrays=[inputX, inputY], axis=-1).astype(np.float32)
        input = input.reshape(n, s, 2)
        return input

    inputX = paddle.to_tensor(data=inputX, dtype="float32").transpose(perm=[1, 0])
    inputY = paddle.to_tensor(data=inputY, dtype="float32").transpose(perm=[1, 0])
    input = paddle.stack(x=[inputX, inputY], axis=-1)
    label = paddle.to_tensor(data=label, dtype="float32").transpose(perm=[1, 0])
    if is_train:
        index = paddle.randperm(n=n)
        index = index[:n]
        input = paddle.index_select(input, index)
        label = paddle.index_select(label, index)
        input = input.reshape([n, s, 2])
    else:
        input = input.reshape([n, s, 2])
    label = label.unsqueeze(axis=-1)
    return input, label, inputPara


def plot(input: np.ndarray, out_pred: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fig_path = osp.join(output_dir, "inference.png")

    xx = np.linspace(-500, 0, 2001)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(input[:, 0], input[:, 1], color="C1", label="Channel geometry")
    plt.plot(input[:, 0], 100 - input[:, 1], color="C1")
    plt.plot(
        xx,
        out_pred,
        "--*",
        color="C2",
        fillstyle="none",
        markevery=len(xx) // 10,
        label="Predicted bacteria distribution",
    )
    plt.xlabel(r"x")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=400)
    plt.close()
    ppsci.utils.logger.info(f"Saving figure to {fig_path}")


def train(cfg: DictConfig):
    # generate training dataset
    inputs_train, labels_train, _ = getdata(**cfg.TRAIN_DATA, is_train=True)

    # set constraints
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": inputs_train},
                "label": {"output": labels_train},
            },
            "batch_size": cfg.TRAIN.batch_size,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        L2RelLoss(reduction="sum"),
        name="sup_constraint",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set model
    model = ppsci.arch.FNO1d(**cfg.MODEL)
    if cfg.TRAIN.use_pretrained_model is True:
        logger.info(
            "Loading pretrained model from {}".format(cfg.TRAIN.pretrained_model_path)
        )
        model.set_state_dict(paddle.load(cfg.TRAIN.pretrained_model_path))

    # set optimizer
    ITERS_PER_EPOCH = int(cfg.TRAIN_DATA.n / cfg.TRAIN.batch_size)
    scheduler = lr_scheduler.Step(
        **cfg.TRAIN.lr_scheduler, iters_per_epoch=ITERS_PER_EPOCH
    )
    optimizer = Adam(scheduler(), weight_decay=cfg.TRAIN.weight_decay)(model)

    # generate test dataset
    inputs_test, labels_test, _ = getdata(**cfg.TEST_DATA, is_train=False)

    # set validator
    l2rel_validator = {
        "validator1": ppsci.validate.SupervisedValidator(
            {
                "dataset": {
                    "name": "NamedArrayDataset",
                    "input": {"input": inputs_test},
                    "label": {"output": labels_test},
                },
                "batch_size": cfg.TRAIN.batch_size,
            },
            L2RelLoss(reduction="sum"),
            metric={"L2Rel": ppsci.metric.L2Rel()},
            name="L2Rel_Validator",
        )
    }

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_with_no_grad=True,
        eval_during_train=cfg.TRAIN.eval_during_train,
        validator=l2rel_validator,
        save_freq=cfg.TRAIN.save_freq,
    )

    # train model
    solver.train()
    # plot losses
    solver.plot_loss_history(by_epoch=True, smooth_step=1)


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.FNO1d(**cfg.MODEL)
    ppsci.utils.save_load.load_pretrain(
        model,
        cfg.EVAL.pretrained_model_path,
    )

    # set data
    x_test, y_test, para = getdata(**cfg.TEST_DATA, is_train=False)
    y_test = y_test.numpy()

    for sample_id in [0, 8]:
        sample, uf, L_p, x1, x2, x3, h = para[:, sample_id]
        mesh = x_test[sample_id, :, :]
        mesh = mesh.numpy()

        y_test_pred = (
            paddle.exp(
                model({"input": x_test[sample_id : sample_id + 1, :, :]})["output"]
            )
            .numpy()
            .flatten()
        )
        logger.info(
            "rel. error is ",
            np.linalg.norm(y_test_pred - y_test[sample_id, :].flatten())
            / np.linalg.norm(y_test[sample_id, :].flatten()),
        )
        xx = np.linspace(-500, 0, 2001)
        plt.figure(figsize=(5, 4))

        plt.plot(mesh[:, 0], mesh[:, 1], color="C1", label="Channel geometry")
        plt.plot(mesh[:, 0], 100 - mesh[:, 1], color="C1")

        plt.plot(
            xx,
            y_test[sample_id, :],
            "--o",
            color="red",
            markevery=len(xx) // 10,
            label="Reference",
        )
        plt.plot(
            xx,
            y_test_pred,
            "--*",
            color="C2",
            fillstyle="none",
            markevery=len(xx) // 10,
            label="Predicted bacteria distribution",
        )

        plt.xlabel(r"x")

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Validation.{sample_id}.pdf")


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.FNO1d(**cfg.MODEL)
    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None, 2001, 2], "float32", name=key)
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy import python_infer

    predictor = python_infer.GeneralPredictor(cfg)

    # evaluate
    input = getdata(**cfg.TEST_DATA, is_train=False, is_inference=True)
    input_dict = {"input": input}

    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_keys = ["output"]
    output_dict = {
        store_key: paddle.exp(paddle.to_tensor(output_dict[infer_key]))
        .numpy()
        .flatten()
        for store_key, infer_key in zip(output_keys, output_dict.keys())
    }

    mesh = input_dict["input"][5, :, :]
    yy = output_dict["output"][5]
    plot(mesh, yy, cfg.output_dir)


@hydra.main(version_base=None, config_path="./conf", config_name="catheter.yaml")
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
            f"cfg.mode should in ['train', 'eval', 'export', 'infer], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

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

from os import path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import paddle
from catheter import FNO1d
from omegaconf import DictConfig

import ppsci
from ppsci.loss import L2RelLoss
from ppsci.optimizer import Adam
from ppsci.optimizer import lr_scheduler
from ppsci.utils import logger


# build data
def getdata(x_path, y_path, para_path, output_path, n_data, n, s, is_train=True):

    # load data
    inputX_raw = np.load(x_path)[:, 0:n_data]
    inputY_raw = np.load(y_path)[:, 0:n_data]
    inputPara_raw = np.load(para_path)[:, 0:n_data]
    output_raw = np.load(output_path)[:, 0:n_data]

    # preprocess data
    inputX = inputX_raw[:, 0::3]
    inputY = inputY_raw[:, 0::3]
    inputPara = inputPara_raw[:, 0::3]
    output = (output_raw[:, 0::3] + output_raw[:, 1::3] + output_raw[:, 2::3]) / 3.0

    inputX = paddle.to_tensor(data=inputX, dtype="float32").transpose(perm=[1, 0])
    inputY = paddle.to_tensor(data=inputY, dtype="float32").transpose(perm=[1, 0])
    input = paddle.stack(x=[inputX, inputY], axis=-1)
    output = paddle.to_tensor(data=output, dtype="float32").transpose(perm=[1, 0])
    if is_train:
        index = paddle.randperm(n=n)
        index = index[:n]

        x = paddle.index_select(input, index)
        y = paddle.index_select(output, index)
        x = x.reshape([n, s, 2])
    else:
        x = input.reshape([n, s, 2])
        y = output

    return x, y, inputPara


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

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
        ppsci.loss.FunctionalLoss(L2RelLoss(reduction="sum")),
        name="sup_constraint",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set model
    model = ppsci.arch.FNO1d(**cfg.MODEL)
    model.set_state_dict(paddle.load(cfg.TRAIN.pretrained_model_path))

    # set optimizer
    ITERS_PER_EPOCH = int(cfg.TRAIN_DATA.n / cfg.TRAIN.batch_size)
    scheduler = lr_scheduler.MultiStepDecay(
        **cfg.TRAIN.lr_scheduler, iters_per_epoch=ITERS_PER_EPOCH
    )
    optimizer = Adam(scheduler, weight_decay=cfg.TRAIN.weight_decay)(model)

    # generate test dataset
    inputs_test, labels_test, _ = getdata(**cfg.TEST_DATA, is_train=False)

    # set validator
    l2rel_validator = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": inputs_test},
                "label": {"output": labels_test},
            },
            "batch_size": cfg.TRAIN.batch_size,
        },
        ppsci.loss.FunctionalLoss(L2RelLoss(reduction="sum")),
        metric={"L2Rel": ppsci.metric.L2Rel()},
        name="L2Rel_Validator",
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=ITERS_PER_EPOCH,
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
    model = FNO1d(**cfg.MODEL)
    model.set_state_dict(paddle.load(cfg.TRAIN.model_path))

    # set data
    x_test, y_test, para = getdata(**cfg.TEST_DATA, is_train=False)
    y_test = y_test.detach().cpu().numpy().flatten()

    for sample_id in [0, 8]:
        sample, uf, L_p, x1, x2, x3, h = para[:, sample_id]
        mesh = x_test[sample_id, :, :]

        y_test_pred = (
            paddle.exp(
                model({"input": x_test[sample_id : sample_id + 1, :, :]})["output"]
            )
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )
        print(
            "rel. error is ",
            np.linalg.norm(y_test_pred - y_test[sample_id, :].numpy())
            / np.linalg.norm(y_test[sample_id, :]),
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
    model = FNO1d(**cfg.MODEL)
    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        cfg=cfg,
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


@hydra.main(version_base=None, config_path="./conf", config_name="catheter.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

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
from omegaconf import DictConfig

import ppsci
from ppsci.loss import L2RelLoss
from ppsci.optimizer import Adam
from ppsci.optimizer import lr_scheduler
from ppsci.utils import logger


# build data
def getdata(x_path, y_path, para_path, output_path, n_data, n, s, is_train=True, is_inference=False):

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
    
    if is_inference:
        return inputX, inputY, inputPara, output
    
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


def transfer(theta):
    L_p = 60 + (250 - 60)/(1 + paddle.exp(theta[0]))
    x1 = -0.5*L_p
    x3 = x1  + 15 + (L_p/2 - 15)/(1 + paddle.exp(theta[2]))
    x2 = -L_p  + L_p/(1 + paddle.exp(theta[1]))
    h = 20   + (10)/(1 + paddle.exp(theta[3]))
    return L_p, x1, x2, x3, h


def inv_transfer(L_p, x2, x3, h):
    x1 = -0.5*L_p
    theta = np.zeros(4)
    theta[0] = np.log( (250 - 60)/(L_p - 60) - 1 )
    theta[1] = np.log( L_p/(x2 + L_p) - 1 )
    theta[2] = np.log( (L_p/2 - 15)/(x3 - x1  - 15) - 1 )
    theta[3] = np.log( 10/(h - 20 ) - 1 )
    return theta


def catheter_mesh_1d_total_length(L_x, L_p, x2, x3, h, N_s):
    x1 = -0.5 * L_p
    n_periods = paddle.floor(x=L_x / L_p)
    L_x_last_period = L_x - n_periods * L_p
    L_p_s = x1 + L_p + (0 - x3) + paddle.sqrt(x=(x2 - x1) ** 2 + h ** 2
        ) + paddle.sqrt(x=(x3 - x2) ** 2 + h ** 2)
    L_s = L_p_s * n_periods + Lx2length(L_x_last_period, L_p, x1, x2, x3, h)
    d_arr = paddle.linspace(start=0, stop=1, num=N_s) * L_s
    period_arr = paddle.floor(x=d_arr / L_p_s).detach()
    d_arr -= period_arr * L_p_s
    xx, yy = d2xy(d_arr, L_p, x1, x2, x3, h)
    xx = xx - period_arr * L_p
    X_Y = paddle.zeros(shape=(1, N_s, 2), dtype='float32')
    X_Y[0, :, 0], X_Y[0, :, 1] = xx, yy
    return X_Y, xx, yy


def Lx2length(L_x, L_p, x1, x2, x3, h):
    l0, l1, l2, l3 = -x3, paddle.sqrt(x=(x2 - x3) ** 2 + h ** 2), paddle.sqrt(x
        =(x1 - x2) ** 2 + h ** 2), L_p + x1
    if L_x < -x3:
        l = L_x
    elif L_x < -x2:
        l = l0 + l1 * (L_x + x3) / (x3 - x2)
    elif L_x < -x1:
        l = l0 + l1 + l2 * (L_x + x2) / (x2 - x1)
    else:
        l = l0 + l1 + l2 + L_x + x1
    return l


def d2xy(d, L_p, x1, x2, x3, h):
    p0, p1, p2, p3 = paddle.to_tensor(data=[0.0, 0.0]).cpu().numpy(), paddle.to_tensor(data
        =[x3, 0.0]).cpu().numpy(), paddle.to_tensor(data=[x2, h]).cpu().numpy(), paddle.to_tensor(data=
        [x1, 0.0]).cpu().numpy()
    v0, v1, v2, v3 = paddle.to_tensor(data=[x3 - 0, 0.0]).cpu().numpy(), paddle.to_tensor(
        data=[x2 - x3, h]).cpu().numpy(), paddle.to_tensor(data=[x1 - x2, -h]
        ).cpu().numpy(), paddle.to_tensor(data=[-L_p - x1, 0.0]).cpu().numpy()
    l0, l1, l2, l3 = -x3.cpu().numpy(), paddle.sqrt(x=(x2 - x3) ** 2 + h ** 2).cpu().numpy(), paddle.sqrt(x
        =(x1 - x2) ** 2 + h ** 2).cpu().numpy(), (L_p + x1).cpu().numpy()
    xx, yy = paddle.zeros(shape=tuple(d.shape)).cpu().numpy(), paddle.zeros(shape=tuple(d
        .shape)).cpu().numpy()
    ind = d < l0
    xx[ind] = d[ind] * v0[0] / l0 + p0[0]
    yy[ind] = d[ind] * v0[1] / l0 + p0[1]
    ind = paddle.logical_and(x=d < l0 + l1, y=d >= l0)
    xx[ind] = (d[ind] - l0) * v1[0] / l1 + p1[0]
    yy[ind] = (d[ind] - l0) * v1[1] / l1 + p1[1]
    ind = paddle.logical_and(x=d < l0 + l1 + l2, y=d >= l0 + l1)
    xx[ind] = (d[ind] - l0 - l1) * v2[0] / l2 + p2[0]
    yy[ind] = (d[ind] - l0 - l1) * v2[1] / l2 + p2[1]
    ind = d >= l0 + l1 + l2
    xx[ind] = (d[ind] - l0 - l1 - l2) * v3[0] / l3 + p3[0]
    yy[ind] = (d[ind] - l0 - l1 - l2) * v3[1] / l3 + p3[1]
    return paddle.to_tensor(xx), paddle.to_tensor(yy)


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
    model = ppsci.arch.FNO1d(**cfg.MODEL)
    ppsci.utils.save_load.load_pretrain(
        model,
        cfg.EVAL.pretrained_model_path,
    )

    # set data
    x_test, y_test, para = getdata(**cfg.TEST_DATA, is_train=False)
    y_test = y_test.numpy().flatten()

    for sample_id in [0, 8]:
        sample, uf, L_p, x1, x2, x3, h = para[:, sample_id]
        mesh = x_test[sample_id, :, :]

        y_test_pred = (
            paddle.exp(
                model({"input": x_test[sample_id : sample_id + 1, :, :]})["output"]
            )
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
    model = ppsci.arch.FNO1d(**cfg.MODEL)
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
    solver.export(input_spec, cfg.INFER.export_path, with_onnx=False)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    # evaluate
    L_p, x2, x3, h = 100.0, -40.0, -30.0, 25.0
    theta0 =  inv_transfer(L_p, x2, x3, h) 
    theta_min = np.copy(theta0)
    theta_min_perturbed = np.copy(theta_min)
    theta = paddle.to_tensor(theta_min_perturbed.astype(np.float32), stop_gradient=False)
    input, output, _ = getdata(**cfg.TEST_DATA, is_train=False)
    input_dict = {"input": input}

    output_dict = predictor.predict(input_dict, cfg.INFER.batch_size)
    # mapping data to cfg.INFER.output_keys
    output_dict = {
        store_key: output_dict[infer_key]
        for store_key, infer_key in zip(['output'], ['intput'])
    }
    
    ppsci.visualize.save_plot_from_1d_dict(
        "./catheter_predict",
        {**input_dict, **output_dict, "output_label": output},
        ("input",),
        ("output", "output_label"),
    )



@hydra.main(version_base=None, config_path="./conf", config_name="catheter.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "inference":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'inference], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

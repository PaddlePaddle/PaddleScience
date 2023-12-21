# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


class Transform:
    def __init__(self, lowb, upb) -> None:
        self.lowb = lowb
        self.upb = upb

    def input_trans(self, input_dict):
        lowb = self.lowb
        upb = self.upb
        for _, v in input_dict.items():
            v = 2.0 * (v - lowb) / (upb - lowb) - 1.0
        return input_dict


# normalization
def dimensionless(lowb, upb, xb_train, yb_train, zb_train, tb_train):
    Xb = np.concatenate([xb_train, yb_train, zb_train, tb_train], 1)
    Xb = 2.0 * (Xb - lowb) / (upb - lowb) - 1.0
    return Xb[:, 0:1], Xb[:, 1:2], Xb[:, 2:3], Xb[:, 3:4]


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet4.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


def train(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # set random seed for reproducibility
    SEED = 1234
    ppsci.utils.misc.set_random_seed(SEED)
    ITERS_PER_EPOCH = cfg.iters_per_epoch
    Re = cfg.re
    N_TRAIN = cfg.ntrain
    NB_TRAIN = cfg.nb_train
    N0_TRAIN = cfg.n0_train
    alpha = cfg.alpha
    beta = cfg.beta

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # Load Data
    data_path = cfg.data_dir
    train_ini1 = np.load(data_path + "train_ini2.npy")
    train_iniv1 = np.load(data_path + "train_iniv2.npy")
    train_xb1 = np.load(data_path + "train_xb2.npy")
    train_vb1 = np.load(data_path + "train_vb2.npy")

    xnode = np.linspace(12.47, 12.66, 191)
    ynode = np.linspace(-1, -0.0031, 998)
    znode = np.linspace(4.61, 4.82, 211)

    x0_train = train_ini1[:, 0:1].astype("float32")
    y0_train = train_ini1[:, 1:2].astype("float32")
    z0_train = train_ini1[:, 2:3].astype("float32")
    t0_train = np.zeros(train_ini1[:, 0:1].shape, np.float32)
    u0_train = train_iniv1[:, 0:1].astype("float32")
    v0_train = train_iniv1[:, 1:2].astype("float32")
    w0_train = train_iniv1[:, 2:3].astype("float32")

    xb_train = train_xb1[:, 0:1].astype("float32")
    yb_train = train_xb1[:, 1:2].astype("float32")
    zb_train = train_xb1[:, 2:3].astype("float32")
    tb_train = train_xb1[:, 3:4].astype("float32")
    ub_train = train_vb1[:, 0:1].astype("float32")
    vb_train = train_vb1[:, 1:2].astype("float32")
    wb_train = train_vb1[:, 2:3].astype("float32")

    x_train1 = xnode.reshape(-1, 1)[np.random.choice(191, 100000, replace=True), :]
    y_train1 = ynode.reshape(-1, 1)[np.random.choice(998, 100000, replace=True), :]
    z_train1 = znode.reshape(-1, 1)[np.random.choice(211, 100000, replace=True), :]
    x_train = np.tile(x_train1, (17, 1)).astype("float32")
    y_train = np.tile(y_train1, (17, 1)).astype("float32")
    z_train = np.tile(z_train1, (17, 1)).astype("float32")

    total_times1 = np.array(list(range(17))) * 0.0065
    t_train1 = total_times1.repeat(100000)
    t_train = t_train1.reshape(-1, 1).astype("float64")
    # Test Data
    test_x = np.load(data_path + "test43_l.npy")
    test_v = np.load(data_path + "test43_vp.npy")
    t = np.array([0.0065, 4 * 0.0065, 7 * 0.0065, 10 * 0.0065, 13 * 0.0065])
    t_star = np.tile(t.reshape(5, 1), (1, 3000)).reshape(-1, 1).astype("float32")
    x_star = np.tile(test_x[:, 0:1], (5, 1)).astype("float32")
    y_star = np.tile(test_x[:, 1:2], (5, 1)).astype("float32")
    z_star = np.tile(test_x[:, 2:3], (5, 1)).astype("float32")
    u_star = test_v[:, 0:1].astype("float32")
    v_star = test_v[:, 1:2].astype("float32")
    w_star = test_v[:, 2:3].astype("float32")
    p_star = test_v[:, 3:4].astype("float32")

    # dimensionless
    Xb = np.concatenate([xb_train, yb_train, zb_train, tb_train], 1)
    lowb = Xb.min(0)  # minimal number in each column
    upb = Xb.max(0)
    trans = Transform(lowb, upb)
    model.register_input_transform(trans.input_trans)

    # set dataloader config
    train_dataloader_cfg_b = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": xb_train, "y": yb_train, "z": zb_train, "t": tb_train},
            "label": {"u": ub_train, "v": vb_train, "w": wb_train},
        },
        "batch_size": NB_TRAIN,
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    train_dataloader_cfg_0 = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x0_train, "y": y0_train, "z": z0_train, "t": t0_train},
            "label": {"u": u0_train, "v": v0_train, "w": w0_train},
        },
        "batch_size": N0_TRAIN,
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    valida_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x_star, "y": y_star, "z": z_star, "t": t_star},
            "label": {"u": u_star, "v": v_star, "w": w_star, "p": p_star},
        },
        "total_size": u_star.shape[0],
        "batch_size": u_star.shape[0],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    geom = ppsci.geometry.PointCloud(
        {"x": x_train, "y": y_train, "z": z_train, "t": t_train}, ("x", "y", "z", "t")
    )
    ## supervised constraint s.t ||u-u_b||
    sup_constraint_b = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_b,
        ppsci.loss.MSELoss("mean", alpha),
        name="Sup_b",
    )

    ## supervised constraint s.t ||u-u_0||
    sup_constraint_0 = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_0,
        ppsci.loss.MSELoss("mean", beta),
        name="Sup_0",
    )

    # set equation constarint s.t. ||F(u)||
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu=1.0 / Re, rho=1.0, dim=3, time=True
        ),
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom,
        {
            "dataset": {"name": "NamedArrayDataset"},
            "batch_size": N_TRAIN,
            "iters_per_epoch": ITERS_PER_EPOCH,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": False,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    constraint = {
        pde_constraint.name: pde_constraint,
        sup_constraint_b.name: sup_constraint_b,
        sup_constraint_0.name: sup_constraint_0,
    }
    # constraint={pde_constraint.name:pde_constraint}
    residual_validator = ppsci.validate.SupervisedValidator(
        valida_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # Wrap validator
    validator = {residual_validator.name: residual_validator}

    # set optimizer
    epoch_list = [250, 4250, 500, 500]
    new_epoch_list = []
    for i, _ in enumerate(epoch_list):
        new_epoch_list.append(sum(epoch_list[: i + 1]))
    EPOCHS = new_epoch_list[-1]
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    lr_scheduler = ppsci.optimizer.lr_scheduler.Piecewise(
        EPOCHS, ITERS_PER_EPOCH, new_epoch_list, lr_list
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        optimizer=optimizer,
        epochs=250,
        lr_scheduler=lr_scheduler,
        iters_per_epoch=ITERS_PER_EPOCH,
        eval_during_train=True,
        log_freq=cfg.log_freq,
        save_freq=100,
        eval_freq=cfg.eval_freq,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=None,
        eval_with_no_grad=False,
        output_dir=OUTPUT_DIR,
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()

    solver.plot_loss_history()


def evaluate(cfg: DictConfig):
    OUTPUT_DIR = cfg.output_dir
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    data_path = cfg.data_dir
    # set random seed for reproducibility
    SEED = cfg.seed
    ppsci.utils.misc.set_random_seed(SEED)

    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # Test Data
    test_x = np.load(data_path + "test43_l.npy")
    test_v = np.load(data_path + "test43_vp.npy")
    t = np.array([0.0065, 4 * 0.0065, 7 * 0.0065, 10 * 0.0065, 13 * 0.0065])
    t_star = paddle.to_tensor(
        np.tile(t.reshape(5, 1), (1, 3000)).reshape(-1, 1).astype("float32")
    )
    x_star = paddle.to_tensor(np.tile(test_x[:, 0:1], (5, 1)).astype("float32"))
    y_star = paddle.to_tensor(np.tile(test_x[:, 1:2], (5, 1)).astype("float32"))
    z_star = paddle.to_tensor(np.tile(test_x[:, 2:3], (5, 1)).astype("float32"))
    u_star = paddle.to_tensor(test_v[:, 0:1].astype("float32"))
    v_star = paddle.to_tensor(test_v[:, 1:2].astype("float32"))
    w_star = paddle.to_tensor(test_v[:, 2:3].astype("float32"))
    p_star = paddle.to_tensor(test_v[:, 3:4].astype("float32"))

    # wrap validator
    checkpoint_path = os.path.join(cfg.pretrained_model_path, "latest.pdparams")
    layer_state_dict = paddle.load(checkpoint_path)
    model.set_state_dict(layer_state_dict)

    # print the relative error
    us = []
    vs = []
    ws = []
    for i in t:
        solution = model(
            {
                "x": x_star,
                "y": y_star,
                "z": z_star,
                "t": t_star[3000 * i : 3000 * (i + 1)],
            }
        )
        u_pred = solution["u"]
        v_pred = solution["v"]
        w_pred = solution["w"]
        p_pred = solution["p"]
        p_pred = p_pred - p_pred.mean() + p_star.mean()
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
        error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
        error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
        us.append(error_u)
        vs.append(error_v)
        ws.append(error_w)
        print(
            "t={:.2f},relative error of u: {:.3e}".format(
                t_star[3000 * i].item(), error_u
            )
        )
        print(
            "t={:.2f},relative error of v: {:.3e}".format(
                t_star[3000 * i].item(), error_v
            )
        )
        print(
            "t={:.2f},relative error of w: {:.3e}".format(
                t_star[3000 * i].item(), error_w
            )
        )
        print(
            "t={:.2f},relative error of p: {:.3e}".format(
                t_star[3000 * i].item(), error_p
            )
        )


if __name__ == "__main__":
    main()

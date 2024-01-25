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

import os.path as osp

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def generate_data(data_dir):
    train_ini1 = np.load(osp.join(data_dir, "train_ini2.npy")).astype(
        paddle.get_default_dtype()
    )
    train_iniv1 = np.load(osp.join(data_dir, "train_iniv2.npy")).astype(
        paddle.get_default_dtype()
    )
    train_xb1 = np.load(osp.join(data_dir, "train_xb2.npy")).astype(
        paddle.get_default_dtype()
    )
    train_vb1 = np.load(osp.join(data_dir, "train_vb2.npy")).astype(
        paddle.get_default_dtype()
    )

    xnode = np.linspace(12.47, 12.66, 191).astype(paddle.get_default_dtype())
    ynode = np.linspace(-1, -0.0031, 998).astype(paddle.get_default_dtype())
    znode = np.linspace(4.61, 4.82, 211).astype(paddle.get_default_dtype())

    x0_train = train_ini1[:, 0:1]
    y0_train = train_ini1[:, 1:2]
    z0_train = train_ini1[:, 2:3]
    t0_train = np.zeros_like(train_ini1[:, 0:1]).astype(paddle.get_default_dtype())
    u0_train = train_iniv1[:, 0:1]
    v0_train = train_iniv1[:, 1:2]
    w0_train = train_iniv1[:, 2:3]

    xb_train = train_xb1[:, 0:1]
    yb_train = train_xb1[:, 1:2]
    zb_train = train_xb1[:, 2:3]
    tb_train = train_xb1[:, 3:4]
    ub_train = train_vb1[:, 0:1]
    vb_train = train_vb1[:, 1:2]
    wb_train = train_vb1[:, 2:3]

    x_train1 = xnode.reshape(-1, 1)[np.random.choice(191, 100000, replace=True), :]
    y_train1 = ynode.reshape(-1, 1)[np.random.choice(998, 100000, replace=True), :]
    z_train1 = znode.reshape(-1, 1)[np.random.choice(211, 100000, replace=True), :]
    x_train = np.tile(x_train1, (17, 1))
    y_train = np.tile(y_train1, (17, 1))
    z_train = np.tile(z_train1, (17, 1))

    total_times1 = (np.array(list(range(17))) * 0.0065).astype(
        paddle.get_default_dtype()
    )
    t_train1 = total_times1.repeat(100000)
    t_train = t_train1.reshape(-1, 1)
    # test data
    test_x = np.load(osp.join(data_dir, "test43_l.npy")).astype(
        paddle.get_default_dtype()
    )
    test_v = np.load(osp.join(data_dir, "test43_vp.npy")).astype(
        paddle.get_default_dtype()
    )
    t = np.array([0.0065, 4 * 0.0065, 7 * 0.0065, 10 * 0.0065, 13 * 0.0065]).astype(
        paddle.get_default_dtype()
    )
    t_star = np.tile(t.reshape(5, 1), (1, 3000)).reshape(-1, 1)
    x_star = np.tile(test_x[:, 0:1], (5, 1))
    y_star = np.tile(test_x[:, 1:2], (5, 1))
    z_star = np.tile(test_x[:, 2:3], (5, 1))
    u_star = test_v[:, 0:1]
    v_star = test_v[:, 1:2]
    w_star = test_v[:, 2:3]
    p_star = test_v[:, 3:4]

    return (
        x_train,
        y_train,
        z_train,
        t_train,
        x0_train,
        y0_train,
        z0_train,
        t0_train,
        u0_train,
        v0_train,
        w0_train,
        xb_train,
        yb_train,
        zb_train,
        tb_train,
        ub_train,
        vb_train,
        wb_train,
        x_star,
        y_star,
        z_star,
        t_star,
        u_star,
        v_star,
        w_star,
        p_star,
    )


class Transform:
    def __init__(self, lowb, upb) -> None:
        self.lowb = {"x": lowb[0], "y": lowb[1], "z": lowb[2], "t": lowb[3]}
        self.upb = {"x": upb[0], "y": upb[1], "z": upb[2], "t": upb[3]}

    def input_trans(self, input_dict):
        for key, v in input_dict.items():
            v = 2.0 * (v - self.lowb[key]) / (self.upb[key] - self.lowb[key]) - 1.0
            input_dict[key] = v
        return input_dict


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # load data
    (
        x_train,
        y_train,
        z_train,
        t_train,
        x0_train,
        y0_train,
        z0_train,
        t0_train,
        u0_train,
        v0_train,
        w0_train,
        xb_train,
        yb_train,
        zb_train,
        tb_train,
        ub_train,
        vb_train,
        wb_train,
        x_star,
        y_star,
        z_star,
        t_star,
        u_star,
        v_star,
        w_star,
        p_star,
    ) = generate_data(cfg.data_dir)

    # normalization
    Xb = np.concatenate([xb_train, yb_train, zb_train, tb_train], 1)
    lowb = Xb.min(0)  # minimal number in each column
    upb = Xb.max(0)
    trans = Transform(paddle.to_tensor(lowb), paddle.to_tensor(upb))
    model.register_input_transform(trans.input_trans)

    # set dataloader config
    train_dataloader_cfg_b = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": xb_train, "y": yb_train, "z": zb_train, "t": tb_train},
            "label": {"u": ub_train, "v": vb_train, "w": wb_train},
        },
        "batch_size": cfg.nb_train,
        "iters_per_epoch": cfg.TRAIN.lr_scheduler.iters_per_epoch,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }

    train_dataloader_cfg_ic = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"x": x0_train, "y": y0_train, "z": z0_train, "t": t0_train},
            "label": {"u": u0_train, "v": v0_train, "w": w0_train},
        },
        "batch_size": cfg.n0_train,
        "iters_per_epoch": cfg.TRAIN.lr_scheduler.iters_per_epoch,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
    }

    valid_dataloader_cfg = {
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
            "shuffle": True,
        },
    }

    geom = ppsci.geometry.PointCloud(
        {"x": x_train, "y": y_train, "z": z_train, "t": t_train}, ("x", "y", "z", "t")
    )
    # supervised constraint s.t ||u-u_b||
    sup_constraint_b = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_b,
        ppsci.loss.MSELoss("mean", cfg.alpha),
        name="Sup_b",
    )

    # supervised constraint s.t ||u-u_0||
    sup_constraint_0 = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_ic,
        ppsci.loss.MSELoss("mean", cfg.beta),
        name="Sup_ic",
    )

    # set equation constarint s.t. ||F(u)||
    equation = {
        "NavierStokes": ppsci.equation.NavierStokes(
            nu=1.0 / cfg.re, rho=1.0, dim=3, time=True
        ),
    }

    pde_constraint = ppsci.constraint.InteriorConstraint(
        equation["NavierStokes"].equations,
        {"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        geom,
        {
            "dataset": {"name": "NamedArrayDataset"},
            "batch_size": cfg.ntrain,
            "iters_per_epoch": cfg.TRAIN.lr_scheduler.iters_per_epoch,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.MSELoss("mean"),
        name="EQ",
    )

    # wrap constraints
    constraint = {
        pde_constraint.name: pde_constraint,
        sup_constraint_b.name: sup_constraint_b,
        sup_constraint_0.name: sup_constraint_0,
    }

    residual_validator = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={"L2R": ppsci.metric.L2Rel()},
        name="Residual",
    )

    # wrap validator
    validator = {residual_validator.name: residual_validator}

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.Piecewise(**cfg.TRAIN.lr_scheduler)()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)
    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=cfg.epochs,
        iters_per_epoch=cfg.TRAIN.lr_scheduler.iters_per_epoch,
        log_freq=cfg.TRAIN.log_freq,
        save_freq=cfg.TRAIN.save_freq,
        eval_freq=cfg.TRAIN.eval_freq,
        eval_during_train=True,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        validator=validator,
        eval_with_no_grad=cfg.TRAIN.eval_with_no_grad,
    )
    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()

    solver.plot_loss_history()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.MLP(**cfg.MODEL)

    # test Data
    test_x = np.load(osp.join(cfg.data_dir, "test43_l.npy")).astype(
        paddle.get_default_dtype()
    )
    test_v = np.load(osp.join(cfg.data_dir, "test43_vp.npy")).astype(
        paddle.get_default_dtype()
    )
    t = np.array([0.0065, 4 * 0.0065, 7 * 0.0065, 10 * 0.0065, 13 * 0.0065]).astype(
        paddle.get_default_dtype()
    )
    t_star = paddle.to_tensor(np.tile(t.reshape(5, 1), (1, 3000)).reshape(-1, 1))
    x_star = paddle.to_tensor(np.tile(test_x[:, 0:1], (5, 1)).reshape(-1, 1))
    y_star = paddle.to_tensor(np.tile(test_x[:, 1:2], (5, 1)).reshape(-1, 1))
    z_star = paddle.to_tensor(np.tile(test_x[:, 2:3], (5, 1)).reshape(-1, 1))
    u_star = paddle.to_tensor(test_v[:, 0:1])
    v_star = paddle.to_tensor(test_v[:, 1:2])
    w_star = paddle.to_tensor(test_v[:, 2:3])
    p_star = paddle.to_tensor(test_v[:, 3:4])

    # wrap validator
    ppsci.utils.load_pretrain(model, cfg.EVAL.pretrained_model_path)

    # print the relative error
    solution = model(
        {
            "x": x_star,
            "y": y_star,
            "z": z_star,
            "t": t_star,
        }
    )
    u_pred = solution["u"].reshape((5, -1))
    v_pred = solution["v"].reshape((5, -1))
    w_pred = solution["w"].reshape((5, -1))
    p_pred = solution["p"].reshape((5, -1))
    u_star = u_star.reshape((5, -1))
    v_star = v_star.reshape((5, -1))
    w_star = w_star.reshape((5, -1))
    p_star = p_star.reshape((5, -1))

    # NS equation can figure out pressure drop, need background pressure p_star.mean()
    p_pred = p_pred - p_pred.mean() + p_star.mean()

    u_error = paddle.linalg.norm(u_pred - u_star, axis=1) / np.linalg.norm(
        u_star, axis=1
    )
    v_error = paddle.linalg.norm(v_pred - v_star, axis=1) / np.linalg.norm(
        v_star, axis=1
    )
    w_error = paddle.linalg.norm(w_pred - w_star, axis=1) / np.linalg.norm(
        w_star, axis=1
    )
    p_error = paddle.linalg.norm(p_pred - p_star, axis=1) / np.linalg.norm(
        w_star, axis=1
    )
    t = np.array([0.0065, 4 * 0.0065, 7 * 0.0065, 10 * 0.0065, 13 * 0.0065])
    plt.plot(t, np.array(u_error))
    plt.plot(t, np.array(v_error))
    plt.plot(t, np.array(w_error))
    plt.plot(t, np.array(p_error))
    plt.legend(["u_error", "v_error", "w_error", "p_error"])
    plt.xlabel("t")
    plt.ylabel("Relative l2 Error")
    plt.title("Relative l2 Error, on test dataset")
    plt.savefig(osp.join(cfg.output_dir, "error.jpg"))
    logger.info("L2 error picture is saved")

    grid_x, grid_y = np.mgrid[
        x_star.min() : x_star.max() : 100j, y_star.min() : y_star.max() : 100j
    ].astype(paddle.get_default_dtype())
    x_plot = paddle.to_tensor(grid_x.reshape(-1, 1))
    y_plot = paddle.to_tensor(grid_y.reshape(-1, 1))
    z_plot = paddle.to_tensor(z_star.min() * paddle.ones(y_plot.shape))
    t_plot = paddle.to_tensor((t[-1]) * np.ones(x_plot.shape), paddle.float32)
    sol = model({"x": x_plot, "y": y_plot, "z": z_plot, "t": t_plot})
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    cmap = plt.cm.get_cmap("jet")

    ax[0].contourf(grid_x, grid_y, sol["u"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[0].set_title("u prediction")
    ax[1].contourf(grid_x, grid_y, sol["v"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[1].set_title("v prediction")
    ax[2].contourf(grid_x, grid_y, sol["w"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[2].set_title("w prediction")
    ax[3].contourf(grid_x, grid_y, sol["p"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[3].set_title("p prediction")
    norm = matplotlib.colors.Normalize(
        vmin=sol["u"].min(), vmax=sol["u"].max()
    )  # set maximum and minimum
    im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax13 = fig.add_axes([0.125, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    ax13 = fig.add_axes([0.325, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    ax13 = fig.add_axes([0.525, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    ax13 = fig.add_axes([0.725, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    plt.savefig(osp.join(cfg.output_dir, "z=0 plane"))

    grid_y, grid_z = np.mgrid[
        y_star.min() : y_star.max() : 100j, z_star.min() : z_star.max() : 100j
    ].astype(paddle.get_default_dtype())
    z_plot = paddle.to_tensor(grid_z.reshape(-1, 1))
    y_plot = paddle.to_tensor(grid_y.reshape(-1, 1))
    x_plot = paddle.to_tensor(x_star.min() * paddle.ones(y_plot.shape))
    t_plot = paddle.to_tensor((t[-1]) * np.ones(x_plot.shape), paddle.float32)
    sol = model({"x": x_plot, "y": y_plot, "z": z_plot, "t": t_plot})
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    cmap = plt.cm.get_cmap("jet")

    ax[0].contourf(grid_y, grid_z, sol["u"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[0].set_title("u prediction")
    ax[1].contourf(grid_y, grid_z, sol["v"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[1].set_title("v prediction")
    ax[2].contourf(grid_y, grid_z, sol["w"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[2].set_title("w prediction")
    ax[3].contourf(grid_y, grid_z, sol["p"].reshape(grid_x.shape), levels=50, cmap=cmap)
    ax[3].set_title("p prediction")
    norm = matplotlib.colors.Normalize(
        vmin=sol["u"].min(), vmax=sol["u"].max()
    )  # set maximum and minimum
    im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax13 = fig.add_axes([0.125, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    ax13 = fig.add_axes([0.325, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    ax13 = fig.add_axes([0.525, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    ax13 = fig.add_axes([0.725, 0.0, 0.175, 0.02])
    plt.colorbar(im, cax=ax13, orientation="horizontal")
    plt.savefig(osp.join(cfg.output_dir, "x=0 plane"))


@hydra.main(version_base=None, config_path="./conf", config_name="VP_NSFNet4.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(
            osp.join("cfg.mode should in ['train', 'eval'], but got", cfg.mode)
        )


if __name__ == "__main__":
    main()

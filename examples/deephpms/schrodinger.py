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

from os import path as osp

import hydra
import numpy as np
import paddle
import paddle.nn.functional as F
import plotting as plot_func
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger
from ppsci.utils import reader
from ppsci.utils import save_load


def pde_loss_func(output_dict, *args):
    losses = F.mse_loss(output_dict["f_pde"], output_dict["du_t"], "sum")
    losses += F.mse_loss(output_dict["g_pde"], output_dict["dv_t"], "sum")
    return {"pde": losses}


def pde_l2_rel_func(output_dict, *args):
    rel_l2_f = paddle.norm(output_dict["du_t"] - output_dict["f_pde"]) / paddle.norm(
        output_dict["du_t"]
    )
    rel_l2_g = paddle.norm(output_dict["dv_t"] - output_dict["g_pde"]) / paddle.norm(
        output_dict["dv_t"]
    )
    metric_dict = {"f_pde_f": rel_l2_f, "f_pde_g": rel_l2_g}
    return metric_dict


def boundary_loss_func(output_dict, *args):
    u_b, v_b = output_dict["u_idn"], output_dict["v_idn"]
    u_lb, u_ub = paddle.split(u_b, 2, axis=0)
    v_lb, v_ub = paddle.split(v_b, 2, axis=0)

    x_b = output_dict["x"]
    du_x = jacobian(u_b, x_b)
    dv_x = jacobian(v_b, x_b)

    du_x_lb, du_x_ub = paddle.split(du_x, 2, axis=0)
    dv_x_lb, dv_x_ub = paddle.split(dv_x, 2, axis=0)

    losses = F.mse_loss(u_lb, u_ub, "sum")
    losses += F.mse_loss(v_lb, v_ub, "sum")
    losses += F.mse_loss(du_x_lb, du_x_ub, "sum")
    losses += F.mse_loss(dv_x_lb, dv_x_ub, "sum")
    return {"boundary": losses}


def sol_l2_rel_func(output_dict, label_dict):
    uv_pred = paddle.sqrt(output_dict["u_idn"] ** 2 + output_dict["v_idn"] ** 2)
    uv_label = paddle.sqrt(label_dict["u_idn"] ** 2 + label_dict["u_idn"] ** 2)
    rel_l2 = paddle.norm(uv_label - uv_pred) / paddle.norm(uv_pred)
    metric_dict = {"uv_sol": rel_l2}
    return metric_dict


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # initialize boundaries
    t_lb = paddle.to_tensor(cfg.T_LB)
    t_ub = paddle.to_tensor(np.pi / cfg.T_UB)
    x_lb = paddle.to_tensor(cfg.X_LB)
    x_ub = paddle.to_tensor(cfg.X_UB)

    # initialize models
    model_idn_u = ppsci.arch.MLP(**cfg.MODEL.idn_u_net)
    model_idn_v = ppsci.arch.MLP(**cfg.MODEL.idn_v_net)
    model_pde_f = ppsci.arch.MLP(**cfg.MODEL.pde_f_net)
    model_pde_g = ppsci.arch.MLP(**cfg.MODEL.pde_g_net)

    # initialize transform
    def transform_uv(_in):
        t, x = _in["t"], _in["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_fg(_in):
        in_idn = {"t": _in["t"], "x": _in["x"]}
        x = _in["x"]
        u = model_idn_u(in_idn)["u_idn"]
        v = model_idn_v(in_idn)["v_idn"]

        du_x = jacobian(u, x)
        du_xx = hessian(u, x)

        dv_x = jacobian(v, x)
        dv_xx = hessian(v, x)

        input_trans = {
            "u": u,
            "v": v,
            "du_x": du_x,
            "dv_x": dv_x,
            "du_xx": du_xx,
            "dv_xx": dv_xx,
        }
        return input_trans

    # register transform
    model_idn_u.register_input_transform(transform_uv)
    model_idn_v.register_input_transform(transform_uv)
    model_pde_f.register_input_transform(transform_fg)
    model_pde_g.register_input_transform(transform_fg)

    # initialize model list
    model_list = ppsci.arch.ModelList(
        (model_idn_u, model_idn_v, model_pde_f, model_pde_g)
    )

    # initialize optimizer
    # Adam
    optimizer_idn = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(
        (model_idn_u, model_idn_v)
    )
    optimizer_pde = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(
        (model_pde_f, model_pde_g)
    )

    # LBFGS
    # optimizer_idn = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)((model_idn_u, model_idn_v))
    # optimizer_pde = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)((model_pde_f, model_pde_g))

    # stage 1: training identification net
    # manually build constraint(s)
    train_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {
                "t": "t_train",
                "x": "x_train",
                "u_idn": "u_train",
                "v_idn": "v_train",
            },
        },
    }

    sup_constraint_idn = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_idn,
        ppsci.loss.MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        name="uv_mse_sup",
    )
    constraint_idn = {sup_constraint_idn.name: sup_constraint_idn}

    # manually build validator
    eval_dataloader_cfg_idn = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "u_idn": "u_star",
                "v_idn": "v_star",
            },
        },
        "batch_size": cfg.TRAIN.batch_size.eval,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    sup_validator_idn = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_idn,
        ppsci.loss.MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        {"l2": ppsci.metric.L2Rel()},
        name="uv_L2_sup",
    )
    validator_idn = {sup_validator_idn.name: sup_validator_idn}

    # initialize solver
    solver = ppsci.solver.Solver(
        model_list,
        constraint_idn,
        cfg.output_dir,
        optimizer_idn,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        validator=validator_idn,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # stage 2: training pde net
    # manually build constraint(s)
    train_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("du_t", "dv_t"),
            "alias_dict": {
                "t": "t_train",
                "x": "x_train",
                "du_t": "t_train",
                "dv_t": "t_train",
            },
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: jacobian(out["v_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
        },
        name="fg_mse_sup",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # manually build validator
    eval_dataloader_cfg_pde = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("du_t", "dv_t"),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "du_t": "t_star",
                "dv_t": "t_star",
            },
        },
        "batch_size": cfg.TRAIN.batch_size.eval,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: jacobian(out["v_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
        },
        {"l2": ppsci.metric.FunctionalMetric(pde_l2_rel_func)},
        name="fg_L2_sup",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # update solver
    solver = ppsci.solver.Solver(
        model_list,
        constraint_pde,
        cfg.output_dir,
        optimizer_pde,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        validator=validator_pde,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # stage 3: training solution net
    # if lbfgs: initialize a new opt with a small initial learning rate in case loss explosion
    # optimizer_idn = ppsci.optimizer.LBFGS(learning_rate=0.01, max_iter=MAX_ITER)(
    #     [model_idn_u, model_idn_v]
    # )
    # manually build constraint(s)
    train_dataloader_cfg_sol_f = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("du_t", "dv_t"),
            "alias_dict": {
                "t": "t_f_train",
                "x": "x_f_train",
                "du_t": "t_f_train",
                "dv_t": "t_f_train",
            },
        },
    }
    train_dataloader_cfg_sol_init = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {"t": "t0", "x": "x0", "u_idn": "u0", "v_idn": "v0"},
        },
    }
    train_dataloader_cfg_sol_bc = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("x",),
            "alias_dict": {"t": "tb", "x": "xb"},
        },
    }

    sup_constraint_sol_f = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_f,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: jacobian(out["v_idn"], out["t"]),
        },
        name="fg_mse_sup",
    )
    sup_constraint_sol_init = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_init,
        ppsci.loss.MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        name="uv0_mse_sup",
    )
    sup_constraint_sol_bc = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_bc,
        ppsci.loss.FunctionalLoss(boundary_loss_func),
        {
            "x": lambda out: out["x"],
            "u_idn": lambda out: out["u_idn"],
            "v_idn": lambda out: out["v_idn"],
        },
        name="uvb_mse_sup",
    )
    constraint_sol = {
        sup_constraint_sol_f.name: sup_constraint_sol_f,
        sup_constraint_sol_init.name: sup_constraint_sol_init,
        sup_constraint_sol_bc.name: sup_constraint_sol_bc,
    }

    # manually build validator
    eval_dataloader_cfg_sol = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "u_idn": "u_star",
                "v_idn": "v_star",
            },
        },
        "batch_size": cfg.TRAIN.batch_size.eval,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    sup_validator_sol = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_sol,
        ppsci.loss.MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        {"l2": ppsci.metric.FunctionalMetric(sol_l2_rel_func)},
        name="uv_L2_sup",
    )
    validator_sol = {sup_validator_sol.name: sup_validator_sol}

    # update solver
    solver = ppsci.solver.Solver(
        model_list,
        constraint_sol,
        cfg.output_dir,
        optimizer_idn,
        None,
        cfg.TRAIN.epochs,
        cfg.TRAIN.iters_per_epoch,
        eval_during_train=cfg.TRAIN.eval_during_train,
        validator=validator_sol,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # initialize boundaries
    t_lb = paddle.to_tensor(cfg.T_LB)
    t_ub = paddle.to_tensor(np.pi / cfg.T_UB)
    x_lb = paddle.to_tensor(cfg.X_LB)
    x_ub = paddle.to_tensor(cfg.X_UB)

    # initialize models
    model_idn_u = ppsci.arch.MLP(**cfg.MODEL.idn_u_net)
    model_idn_v = ppsci.arch.MLP(**cfg.MODEL.idn_v_net)
    model_pde_f = ppsci.arch.MLP(**cfg.MODEL.pde_f_net)
    model_pde_g = ppsci.arch.MLP(**cfg.MODEL.pde_g_net)

    # initialize transform
    def transform_uv(_in):
        t, x = _in["t"], _in["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_fg(_in):
        in_idn = {"t": _in["t"], "x": _in["x"]}
        x = _in["x"]
        u = model_idn_u(in_idn)["u_idn"]
        v = model_idn_v(in_idn)["v_idn"]

        du_x = jacobian(u, x)
        du_xx = hessian(u, x)

        dv_x = jacobian(v, x)
        dv_xx = hessian(v, x)

        input_trans = {
            "u": u,
            "v": v,
            "du_x": du_x,
            "dv_x": dv_x,
            "du_xx": du_xx,
            "dv_xx": dv_xx,
        }
        return input_trans

    # register transform
    model_idn_u.register_input_transform(transform_uv)
    model_idn_v.register_input_transform(transform_uv)
    model_pde_f.register_input_transform(transform_fg)
    model_pde_g.register_input_transform(transform_fg)

    # initialize model list
    model_list = ppsci.arch.ModelList(
        (model_idn_u, model_idn_v, model_pde_f, model_pde_g)
    )
    # stage 3: solution net
    # load pretrained model
    save_load.load_pretrain(model_list, cfg.EVAL.pretrained_model_path)

    # load pretrained model
    save_load.load_pretrain(model_list, cfg.EVAL.pretrained_model_path)

    # load dataset
    dataset_val = reader.load_mat_file(
        cfg.DATASET_PATH_SOL,
        keys=("t", "x", "uv_sol", "u_sol", "v_sol"),
        alias_dict={
            "t": "t_ori",
            "x": "x_ori",
            "uv_sol": "Exact_uv_ori",
            "u_sol": "u_star",
            "v_sol": "v_star",
        },
    )

    t_sol, x_sol = np.meshgrid(
        np.squeeze(dataset_val["t"]), np.squeeze(dataset_val["x"])
    )
    t_sol_flatten = paddle.to_tensor(
        t_sol.flatten()[:, None], dtype=paddle.get_default_dtype(), stop_gradient=False
    )
    x_sol_flatten = paddle.to_tensor(
        x_sol.flatten()[:, None], dtype=paddle.get_default_dtype(), stop_gradient=False
    )
    pred = model_list({"t": t_sol_flatten, "x": x_sol_flatten})
    u_sol_pred = pred["u_idn"].numpy()
    v_sol_pred = pred["v_idn"].numpy()
    uv_sol_pred = np.sqrt(u_sol_pred**2 + v_sol_pred**2)

    # eval
    uv_sol_star = np.sqrt(dataset_val["u_sol"] ** 2 + dataset_val["v_sol"] ** 2)
    error_uv = np.linalg.norm(uv_sol_star - uv_sol_pred, 2) / np.linalg.norm(
        uv_sol_star, 2
    )
    logger.info(f"l2_error_uv: {error_uv}")

    # plotting
    plot_points = paddle.concat([t_sol_flatten, x_sol_flatten], axis=-1).numpy()
    plot_func.draw_and_save(
        figname="schrodinger_uv_sol",
        data_exact=dataset_val["uv_sol"],
        data_learned=uv_sol_pred,
        boundary=[cfg.T_LB, cfg.T_UB, cfg.X_LB, cfg.X_UB],
        griddata_points=plot_points,
        griddata_xi=(t_sol, x_sol),
        save_path=cfg.output_dir,
    )


def export(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # 设置使用CPU
    paddle.set_device("cpu")
    print(f"Using device: {paddle.get_device()}")

    # initialize boundaries
    t_lb = paddle.to_tensor(cfg.T_LB)
    t_ub = paddle.to_tensor(np.pi / cfg.T_UB)
    x_lb = paddle.to_tensor(cfg.X_LB)
    x_ub = paddle.to_tensor(cfg.X_UB)

    # initialize models
    model_idn_u = ppsci.arch.MLP(**cfg.MODEL.idn_u_net)
    model_idn_v = ppsci.arch.MLP(**cfg.MODEL.idn_v_net)

    # initialize transform
    def transform_uv(_in):
        t, x = _in["t"], _in["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    # register transform
    model_idn_u.register_input_transform(transform_uv)
    model_idn_v.register_input_transform(transform_uv)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn_u, model_idn_v))

    # 加载预训练模型
    save_load.load_pretrain(model_list, cfg.INFER.pretrained_model_path)
    model_list.eval()

    # 确保导出目录存在
    import json
    import os

    export_dir = os.path.dirname(cfg.INFER.export_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    # 保存模型参数
    params_save_path = cfg.INFER.export_path + "_params.pdparams"
    paddle.save(model_list.state_dict(), params_save_path)

    # 保存模型结构信息
    model_info = {
        "model_type": "PaddleScience_Schrodinger",
        "input_keys": ["t", "x"],
        "output_keys": ["u_idn", "v_idn"],
        "boundaries": {
            "T_LB": float(cfg.T_LB),
            "T_UB": float(cfg.T_UB),
            "X_LB": float(cfg.X_LB),
            "X_UB": float(cfg.X_UB),
        },
    }

    info_save_path = cfg.INFER.export_path + "_info.json"
    with open(info_save_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Model exported to {params_save_path} and {info_save_path}")


def inference(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # initialize boundaries
    t_lb = paddle.to_tensor(cfg.T_LB)
    t_ub = paddle.to_tensor(np.pi / cfg.T_UB)
    x_lb = paddle.to_tensor(cfg.X_LB)
    x_ub = paddle.to_tensor(cfg.X_UB)

    # initialize models - 只初始化需要的模型
    model_idn_u = ppsci.arch.MLP(**cfg.MODEL.idn_u_net)
    model_idn_v = ppsci.arch.MLP(**cfg.MODEL.idn_v_net)

    # initialize transform
    def transform_uv(_in):
        t, x = _in["t"], _in["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    # register transform
    model_idn_u.register_input_transform(transform_uv)
    model_idn_v.register_input_transform(transform_uv)

    # initialize model list - 只包含需要的模型
    model_list = ppsci.arch.ModelList((model_idn_u, model_idn_v))

    # load pretrained model
    save_load.load_pretrain(model_list, cfg.INFER.pretrained_model_path)

    # 尝试加载数据集以获得与eval相同的网格点
    try:
        dataset_path = getattr(cfg, "DATASET_PATH_SOL", "./datasets/NLS.mat")
        if not osp.exists(dataset_path):
            dataset_path = "./datasets/NLS.mat"

        dataset_val = reader.load_mat_file(
            dataset_path,
            keys=("t", "x", "uv_sol", "u_sol", "v_sol"),
            alias_dict={
                "t": "t_ori",
                "x": "x_ori",
                "uv_sol": "Exact_uv_ori",
                "u_sol": "u_star",
                "v_sol": "v_star",
            },
        )

        # 使用数据集中的网格点
        t_mesh, x_mesh = np.meshgrid(
            np.squeeze(dataset_val["t"]), np.squeeze(dataset_val["x"])
        )

    except Exception:
        # 回退到默认网格
        t_points = np.linspace(cfg.T_LB, cfg.T_UB, cfg.INFER.t_points)
        x_points = np.linspace(cfg.X_LB, cfg.X_UB, cfg.INFER.x_points)
        t_mesh, x_mesh = np.meshgrid(t_points, x_points)
        dataset_val = None

    t_flatten = paddle.to_tensor(
        t_mesh.flatten()[:, None], dtype=paddle.get_default_dtype(), stop_gradient=False
    )
    x_flatten = paddle.to_tensor(
        x_mesh.flatten()[:, None], dtype=paddle.get_default_dtype(), stop_gradient=False
    )

    # 使用模型进行预测
    pred = model_list({"t": t_flatten, "x": x_flatten})
    u_pred = pred["u_idn"].numpy()
    v_pred = pred["v_idn"].numpy()
    uv_pred = np.sqrt(u_pred**2 + v_pred**2)

    # 保存结果
    np.savez(
        osp.join(cfg.output_dir, "inference_results.npz"),
        t=t_mesh.flatten(),
        x=x_mesh.flatten(),
        u=u_pred,
        v=v_pred,
        uv=uv_pred,
    )

    # 可视化
    plot_points = paddle.concat([t_flatten, x_flatten], axis=-1).numpy()
    data_exact = (
        dataset_val["uv_sol"] if dataset_val is not None else np.zeros_like(uv_pred)
    )
    figname = (
        "schrodinger_uv_infer_with_exact"
        if dataset_val is not None
        else "schrodinger_uv_infer"
    )

    plot_func.draw_and_save(
        figname=figname,
        data_exact=data_exact,
        data_learned=uv_pred,
        boundary=[cfg.T_LB, cfg.T_UB, cfg.X_LB, cfg.X_UB],
        griddata_points=plot_points,
        griddata_xi=(t_mesh, x_mesh),
        save_path=cfg.output_dir,
    )

    # 计算误差（如果有真实解）
    if dataset_val is not None:
        uv_exact = np.sqrt(dataset_val["u_sol"] ** 2 + dataset_val["v_sol"] ** 2)
        error_uv = np.linalg.norm(uv_exact - uv_pred, 2) / np.linalg.norm(uv_exact, 2)
        logger.info(f"Inference L2 error (vs exact solution): {error_uv}")

    logger.info(f"Inference completed. Results saved to {cfg.output_dir}")


@hydra.main(version_base=None, config_path="./conf", config_name="schrodinger.yaml")
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

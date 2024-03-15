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
    losses = F.mse_loss(output_dict["f_pde"], output_dict["dw_t"], "sum")
    return losses


def pde_l2_rel_func(output_dict, *args):
    rel_l2 = paddle.norm(output_dict["dw_t"] - output_dict["f_pde"]) / paddle.norm(
        output_dict["dw_t"]
    )
    metric_dict = {"f_pde": rel_l2}
    return metric_dict


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # initialize boundaries
    # t, x, y
    lb = paddle.to_tensor(list(cfg.LB))
    ub = paddle.to_tensor(list(cfg.UB))

    # initialize models
    model_idn = ppsci.arch.MLP(**cfg.MODEL.idn_net)
    model_pde = ppsci.arch.MLP(**cfg.MODEL.pde_net)

    # initialize transform
    def transform_w(_in):
        t, x, y = _in["t"], _in["x"], _in["y"]
        X = paddle.concat([t, x, y], axis=1)
        H = 2.0 * (X - lb) * paddle.pow((ub - lb), -1) - 1.0
        t, x, y = paddle.split(H, 3, axis=1)
        input_trans = {"t": t, "x": x, "y": y}
        return input_trans

    def transform_f(_in):
        in_idn = {"t": _in["t"], "x": _in["x"], "y": _in["y"]}
        x, y = _in["x"], _in["y"]
        w = model_idn(in_idn)["w_idn"]
        dw_x = jacobian(w, x)
        dw_y = jacobian(w, y)

        dw_xx = hessian(w, x)
        dw_yy = hessian(w, y)
        dw_xy = jacobian(dw_x, y)

        input_trans = {
            "u": _in["u"],
            "v": _in["v"],
            "w": w,
            "dw_x": dw_x,
            "dw_y": dw_y,
            "dw_xx": dw_xx,
            "dw_xy": dw_xy,
            "dw_yy": dw_yy,
        }
        return input_trans

    # register transform
    model_idn.register_input_transform(transform_w)
    model_pde.register_input_transform(transform_f)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn, model_pde))

    # initialize optimizer
    # Adam
    optimizer_idn = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model_idn)
    optimizer_pde = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model_pde)

    # LBFGS
    # optimizer_idn = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)(model_idn)
    # optimizer_pde = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)(model_pde)

    # stage 1: training identification net
    # manually build constraint(s)
    train_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("w_idn",),
            "alias_dict": {
                "t": "t_train",
                "x": "x_train",
                "y": "y_train",
                "u": "u_train",
                "v": "v_train",
                "w_idn": "w_train",
            },
        },
    }

    sup_constraint_idn = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_idn,
        ppsci.loss.MSELoss("sum"),
        {"w_idn": lambda out: out["w_idn"]},
        name="w_mse_sup",
    )
    constraint_idn = {sup_constraint_idn.name: sup_constraint_idn}

    # manually build validator
    eval_dataloader_cfg_idn = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("w_idn",),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "y": "y_star",
                "u": "u_star",
                "v": "v_star",
                "w_idn": "w_star",
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
        {"w_idn": lambda out: out["w_idn"]},
        {"l2": ppsci.metric.L2Rel()},
        name="w_L2_sup",
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
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("dw_t",),
            "alias_dict": {
                "t": "t_train",
                "x": "x_train",
                "y": "y_train",
                "u": "u_train",
                "v": "v_train",
                "dw_t": "t_train",
            },
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "dw_t": lambda out: jacobian(out["w_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
        },
        name="f_mse_sup",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # manually build validator
    eval_dataloader_cfg_pde = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("dw_t",),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "y": "y_star",
                "u": "u_star",
                "v": "v_star",
                "dw_t": "t_star",
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
            "dw_t": lambda out: jacobian(out["w_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
        },
        {"l2": ppsci.metric.FunctionalMetric(pde_l2_rel_func)},
        name="f_L2_sup",
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

    # stage 3: training solution net, reuse identification net
    # manually build constraint(s)
    train_dataloader_cfg_sol_f = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("dw_t",),
            "alias_dict": {
                "t": "t_f_train",
                "x": "x_f_train",
                "y": "y_f_train",
                "u": "u_f_train",
                "v": "v_f_train",
                "dw_t": "t_f_train",
            },
        },
    }
    train_dataloader_cfg_sol_bc = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("wb_sol",),
            "alias_dict": {
                "t": "tb",
                "x": "xb",
                "y": "yb",
                "wb_sol": "wb",
                "u": "xb",
                "v": "yb",
            },
        },
    }

    sup_constraint_sol_f = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_f,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "f_pde": lambda out: out["f_pde"],
            "dw_t": lambda out: jacobian(out["w_idn"], out["t"]),
        },
        name="f_mse_sup",
    )
    sup_constraint_sol_bc = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_bc,
        ppsci.loss.MSELoss("sum"),
        {"wb_sol": lambda out: out["w_idn"]},
        name="ub_mse_sup",
    )
    constraint_sol = {
        sup_constraint_sol_f.name: sup_constraint_sol_f,
        sup_constraint_sol_bc.name: sup_constraint_sol_bc,
    }

    # manually build validator
    eval_dataloader_cfg_sol = {
        "dataset": {
            "name": "MatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x", "y", "u", "v"),
            "label_keys": ("w_sol",),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "y": "y_star",
                "w_sol": "w_star",
                "u": "u_star",
                "v": "v_star",
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
        {"w_sol": lambda out: out["w_idn"]},
        {"l2": ppsci.metric.L2Rel()},
        name="w_L2_sup",
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
    # t, x, y
    lb = paddle.to_tensor(list(cfg.LB))
    ub = paddle.to_tensor(list(cfg.UB))

    # initialize models
    model_idn = ppsci.arch.MLP(**cfg.MODEL.idn_net)
    model_pde = ppsci.arch.MLP(**cfg.MODEL.pde_net)

    # initialize transform
    def transform_w(_in):
        t, x, y = _in["t"], _in["x"], _in["y"]
        X = paddle.concat([t, x, y], axis=1)
        H = 2.0 * (X - lb) * paddle.pow((ub - lb), -1) - 1.0
        t, x, y = paddle.split(H, 3, axis=1)
        input_trans = {"t": t, "x": x, "y": y}
        return input_trans

    def transform_f(_in):
        in_idn = {"t": _in["t"], "x": _in["x"], "y": _in["y"]}
        x, y = _in["x"], _in["y"]
        w = model_idn(in_idn)["w_idn"]
        dw_x = jacobian(w, x)
        dw_y = jacobian(w, y)

        dw_xx = hessian(w, x)
        dw_yy = hessian(w, y)
        dw_xy = jacobian(dw_x, y)

        input_trans = {
            "u": _in["u"],
            "v": _in["v"],
            "w": w,
            "dw_x": dw_x,
            "dw_y": dw_y,
            "dw_xx": dw_xx,
            "dw_xy": dw_xy,
            "dw_yy": dw_yy,
        }
        return input_trans

    # register transform
    model_idn.register_input_transform(transform_w)
    model_pde.register_input_transform(transform_f)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn, model_pde))

    # stage 3: solution net
    # load pretrained model
    save_load.load_pretrain(model_list, cfg.EVAL.pretrained_model_path)

    # load pretrained model
    save_load.load_pretrain(model_list, cfg.EVAL.pretrained_model_path)

    # load dataset
    dataset_val = reader.load_mat_file(
        cfg.DATASET_PATH_SOL,
        keys=("t", "x", "y", "w_sol", "grid_data"),
        alias_dict={
            "t": "t_star",
            "x": "x_star",
            "y": "y_star",
            "w_sol": "w_star",
            "grid_data": "X_star",
        },
    )
    input_dict = {
        "t": paddle.to_tensor(
            dataset_val["t"], dtype=paddle.get_default_dtype(), stop_gradient=False
        ),
        "x": paddle.to_tensor(
            dataset_val["x"], dtype=paddle.get_default_dtype(), stop_gradient=False
        ),
        "y": paddle.to_tensor(
            dataset_val["y"], dtype=paddle.get_default_dtype(), stop_gradient=False
        ),
    }

    w_sol_pred = model_idn(input_dict)

    # eval
    l2_error = np.linalg.norm(
        dataset_val["w_sol"] - w_sol_pred["w_idn"], 2
    ) / np.linalg.norm(
        dataset_val["w_sol"], 2
    )  # stage 1&3 use the same net in this example
    logger.info(f"l2_error: {l2_error}")

    # plotting
    plot_func.draw_and_save_ns(
        figname="navier_stokes_sol",
        data_exact=dataset_val["w_sol"].reshape([-1, 151]),
        data_learned=w_sol_pred["w_idn"].reshape([-1, 151]).numpy(),
        grid_data=dataset_val["grid_data"].reshape([-1, 2]),
        save_path=cfg.output_dir,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="navier_stokes.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

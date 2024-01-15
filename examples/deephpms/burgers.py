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
    return losses


def pde_l2_rel_func(output_dict, *args):
    rel_l2 = paddle.norm(output_dict["du_t"] - output_dict["f_pde"]) / paddle.norm(
        output_dict["du_t"]
    )
    metric_dict = {"f_pde": rel_l2}
    return metric_dict


def boundary_loss_func(output_dict, *args):
    u_b = output_dict["u_sol"]
    u_lb, u_ub = paddle.split(u_b, 2, axis=0)

    x_b = output_dict["x"]
    du_x = jacobian(u_b, x_b)

    du_x_lb, du_x_ub = paddle.split(du_x, 2, axis=0)

    losses = F.mse_loss(u_lb, u_ub, "sum")
    losses += F.mse_loss(du_x_lb, du_x_ub, "sum")
    return losses


def train(cfg: DictConfig):
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # initialize burgers boundaries
    t_lb = paddle.to_tensor(cfg.T_LB)
    t_ub = paddle.to_tensor(cfg.T_UB)
    x_lb = paddle.to_tensor(cfg.X_LB)
    x_ub = paddle.to_tensor(cfg.T_UB)

    # initialize models
    model_idn = ppsci.arch.MLP(**cfg.MODEL.idn_net)
    model_pde = ppsci.arch.MLP(**cfg.MODEL.pde_net)
    model_sol = ppsci.arch.MLP(**cfg.MODEL.sol_net)

    # initialize transform
    def transform_u(_in):
        t, x = _in["t"], _in["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_f(input, model, out_key):
        in_idn = {"t": input["t"], "x": input["x"]}
        x = input["x"]
        u = model(in_idn)[out_key]
        du_x = jacobian(u, x)
        du_xx = hessian(u, x)
        input_trans = {"u_x": u, "du_x": du_x, "du_xx": du_xx}
        return input_trans

    def transform_f_idn(_in):
        return transform_f(_in, model_idn, "u_idn")

    def transform_f_sol(_in):
        return transform_f(_in, model_sol, "u_sol")

    # register transform
    model_idn.register_input_transform(transform_u)
    model_pde.register_input_transform(transform_f_idn)
    model_sol.register_input_transform(transform_u)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn, model_pde, model_sol))

    # initialize optimizer
    # Adam
    optimizer_idn = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model_idn)
    optimizer_pde = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model_pde)
    optimizer_sol = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(model_sol)

    # LBFGS
    # optimizer_idn = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)(model_idn)
    # optimizer_pde = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)(model_pde)
    # optimizer_sol = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)(model_sol)

    # stage 1: training identification net
    # manually build constraint(s)
    train_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn",),
            "alias_dict": {"t": "t_train", "x": "x_train", "u_idn": "u_train"},
        },
    }

    sup_constraint_idn = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_idn,
        ppsci.loss.MSELoss("sum"),
        {"u_idn": lambda out: out["u_idn"]},
        name="u_mse_sup",
    )
    constraint_idn = {sup_constraint_idn.name: sup_constraint_idn}

    # manually build validator
    eval_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn",),
            "alias_dict": {"t": "t_star", "x": "x_star", "u_idn": "u_star"},
        },
    }

    sup_validator_idn = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_idn,
        ppsci.loss.MSELoss("sum"),
        {"u_idn": lambda out: out["u_idn"]},
        {"l2": ppsci.metric.L2Rel()},
        name="u_L2_sup",
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
            "label_keys": ("du_t",),
            "alias_dict": {"t": "t_train", "x": "x_train", "du_t": "t_train"},
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
        },
        name="f_mse_sup",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # manually build validator
    eval_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("du_t",),
            "alias_dict": {"t": "t_star", "x": "x_star", "du_t": "t_star"},
        },
    }

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(pde_loss_func),
        {
            "du_t": lambda out: jacobian(out["u_idn"], out["t"]),
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

    # stage 3: training solution net
    # re-register transform for model 2, fit for loss of stage 3
    model_pde.register_input_transform(transform_f_sol)

    # manually build constraint(s)
    train_dataloader_cfg_sol_f = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("du_t",),
            "alias_dict": {"t": "t_f_train", "x": "x_f_train", "du_t": "t_f_train"},
        },
    }
    train_dataloader_cfg_sol_init = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_sol",),
            "alias_dict": {"t": "t0", "x": "x0", "u_sol": "u0"},
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
            "du_t": lambda out: jacobian(out["u_sol"], out["t"]),
        },
        name="f_mse_sup",
    )
    sup_constraint_sol_init = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_init,
        ppsci.loss.MSELoss("sum"),
        {"u_sol": lambda out: out["u_sol"]},
        name="u0_mse_sup",
    )
    sup_constraint_sol_bc = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_bc,
        ppsci.loss.FunctionalLoss(boundary_loss_func),
        {
            "x": lambda out: out["x"],
            "u_sol": lambda out: out["u_sol"],
        },
        name="ub_mse_sup",
    )
    constraint_sol = {
        sup_constraint_sol_f.name: sup_constraint_sol_f,
        sup_constraint_sol_init.name: sup_constraint_sol_init,
        sup_constraint_sol_bc.name: sup_constraint_sol_bc,
    }

    # manually build validator
    eval_dataloader_cfg_sol = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": cfg.DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_sol",),
            "alias_dict": {"t": "t_star", "x": "x_star", "u_sol": "u_star"},
        },
    }

    sup_validator_sol = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_sol,
        ppsci.loss.MSELoss("sum"),
        {"u_sol": lambda out: out["u_sol"]},
        {"l2": ppsci.metric.L2Rel()},
        name="u_L2_sup",
    )
    validator_sol = {sup_validator_sol.name: sup_validator_sol}

    # update solver
    solver = ppsci.solver.Solver(
        model_list,
        constraint_sol,
        cfg.output_dir,
        optimizer_sol,
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

    # initialize burgers boundaries
    t_lb = paddle.to_tensor(cfg.T_LB)
    t_ub = paddle.to_tensor(cfg.T_UB)
    x_lb = paddle.to_tensor(cfg.X_LB)
    x_ub = paddle.to_tensor(cfg.T_UB)

    # initialize models
    model_idn = ppsci.arch.MLP(**cfg.MODEL.idn_net)
    model_pde = ppsci.arch.MLP(**cfg.MODEL.pde_net)
    model_sol = ppsci.arch.MLP(**cfg.MODEL.sol_net)

    # initialize transform
    def transform_u(_in):
        t, x = _in["t"], _in["x"]
        t = 2.0 * (t - t_lb) * paddle.pow((t_ub - t_lb), -1) - 1.0
        x = 2.0 * (x - x_lb) * paddle.pow((x_ub - x_lb), -1) - 1.0
        input_trans = {"t": t, "x": x}
        return input_trans

    def transform_f(input, model, out_key):
        in_idn = {"t": input["t"], "x": input["x"]}
        x = input["x"]
        u = model(in_idn)[out_key]
        du_x = jacobian(u, x)
        du_xx = hessian(u, x)
        input_trans = {"u_x": u, "du_x": du_x, "du_xx": du_xx}
        return input_trans

    def transform_f_sol(_in):
        return transform_f(_in, model_sol, "u_sol")

    # register transform
    model_idn.register_input_transform(transform_u)
    model_pde.register_input_transform(transform_f_sol)
    model_sol.register_input_transform(transform_u)

    # initialize model list
    model_list = ppsci.arch.ModelList((model_idn, model_pde, model_sol))

    # stage 3: solution net
    # load pretrained model
    save_load.load_pretrain(model_list, cfg.EVAL.pretrained_model_path)

    # load dataset
    dataset_val = reader.load_mat_file(
        cfg.DATASET_PATH_SOL,
        keys=("t", "x", "u_sol"),
        alias_dict={
            "t": "t_ori",
            "x": "x_ori",
            "u_sol": "Exact_ori",
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
    u_sol_pred = model_list({"t": t_sol_flatten, "x": x_sol_flatten})

    # eval
    l2_error = np.linalg.norm(
        dataset_val["u_sol"] - u_sol_pred["u_sol"], 2
    ) / np.linalg.norm(dataset_val["u_sol"], 2)
    logger.info(f"l2_error: {l2_error}")

    # plotting
    plot_points = paddle.concat([t_sol_flatten, x_sol_flatten], axis=-1).numpy()
    plot_func.draw_and_save(
        figname="burgers_sol",
        data_exact=dataset_val["u_sol"],
        data_learned=u_sol_pred["u_sol"].numpy(),
        boundary=[cfg.T_LB, cfg.T_UB, cfg.X_LB, cfg.X_UB],
        griddata_points=plot_points,
        griddata_xi=(t_sol, x_sol),
        save_path=cfg.output_dir,
    )


@hydra.main(version_base=None, config_path="./conf", config_name="burgers.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

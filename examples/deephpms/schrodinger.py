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

import numpy as np
import paddle
import paddle.nn.functional as F

import ppsci
from ppsci.utils import logger


def pde_loss_compute(output_dict):
    losses = F.mse_loss(output_dict["f_pde"], output_dict["du_t"], "sum")
    losses += F.mse_loss(output_dict["g_pde"], output_dict["dv_t"], "sum")
    return losses


def pde_l2_rel_compute(output_dict):
    rel_l2_f = paddle.norm(output_dict["du_t"] - output_dict["f_pde"]) / paddle.norm(
        output_dict["du_t"]
    )
    rel_l2_g = paddle.norm(output_dict["dv_t"] - output_dict["g_pde"]) / paddle.norm(
        output_dict["dv_t"]
    )
    metric_dict = {"f_pde": rel_l2_f, "f_pde": rel_l2_g}
    return metric_dict


def boundary_loss_compute(output_dict):
    u_b, v_b = output_dict["u_idn"], output_dict["v_idn"]
    u_lb, u_ub = paddle.split(u_b, 2, axis=0)
    v_lb, v_ub = paddle.split(v_b, 2, axis=0)

    x_b = output_dict["x"]
    du_x = ppsci.autodiff.jacobian(u_b, x_b)
    dv_x = ppsci.autodiff.jacobian(v_b, x_b)

    du_x_lb, du_x_ub = paddle.split(du_x, 2, axis=0)
    dv_x_lb, dv_x_ub = paddle.split(dv_x, 2, axis=0)

    losses = F.mse_loss(u_lb, u_ub, "sum")
    losses += F.mse_loss(v_lb, v_ub, "sum")
    losses += F.mse_loss(du_x_lb, du_x_ub, "sum")
    losses += F.mse_loss(dv_x_lb, dv_x_ub, "sum")
    return losses


if __name__ == "__main__":
    EPOCHS = 50000  # 1 for LBFGS
    MAX_ITER = 50000  # for LBFGS
    ITERS_PER_EPOCH = 1
    DATASET_PATH = "./datasets/DeepHPMs/NLS.mat"
    DATASET_PATH_SOL = "./datasets/DeepHPMs/NLS.mat"
    OUTPUT_DIR = "./outs/"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize boundaries
    t_lb = paddle.to_tensor([0.0])
    t_ub = paddle.to_tensor([np.pi / 2.0])
    x_lb = paddle.to_tensor([-5.0])
    x_ub = paddle.to_tensor([5.0])

    # initialize models
    model_idn_u = ppsci.arch.MLP(("t", "x"), ("u_idn",), 4, 50, "sin")
    model_idn_v = ppsci.arch.MLP(("t", "x"), ("v_idn",), 4, 50, "sin")
    model_pde_f = ppsci.arch.MLP(
        ("u", "v", "du_x", "dv_x", "du_xx", "dv_xx"),
        ("f_pde",),
        2,
        100,
        "sin",
        input_dim=6,
    )
    model_pde_g = ppsci.arch.MLP(
        ("u", "v", "du_x", "dv_x", "du_xx", "dv_xx"),
        ("g_pde",),
        2,
        100,
        "sin",
    )

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

        du_x = ppsci.autodiff.jacobian(u, x)
        du_xx = ppsci.autodiff.hessian(u, x)

        dv_x = ppsci.autodiff.jacobian(v, x)
        dv_xx = ppsci.autodiff.hessian(v, x)

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
    optimizer_idn = ppsci.optimizer.Adam(1e-4)((model_idn_u, model_idn_v))
    optimizer_pde = ppsci.optimizer.Adam(1e-4)((model_pde_f, model_pde_g))

    # LBFGS
    # optimizer_idn = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)((model_idn_u, model_idn_v))
    # optimizer_pde = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)((model_pde_f, model_pde_g))

    # stage 1: training identification net
    # maunally build constraint(s)
    train_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
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

    # maunally build validator
    eval_dataloader_cfg_idn = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "u_idn": "u_star",
                "v_idn": "v_star",
            },
        },
    }

    sup_validator_idn = ppsci.validate.SupervisedValidator(
        train_dataloader_cfg_idn,
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
        OUTPUT_DIR,
        optimizer_idn,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_idn,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # stage 2: training pde net
    # maunally build constraint(s)
    train_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
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
        ppsci.loss.FunctionalLoss(pde_loss_compute),
        {
            "du_t": lambda out: ppsci.autodiff.jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: ppsci.autodiff.jacobian(out["v_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
        },
        name="fg_mse_sup",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    # maunally build validator
    eval_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("t", "x"),
            "label_keys": ("du_t", "dv_t"),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "du_t": "t_star",
                "dv_t": "t_star",
            },
        },
    }

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(pde_loss_compute),
        {
            "du_t": lambda out: ppsci.autodiff.jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: ppsci.autodiff.jacobian(out["v_idn"], out["t"]),
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
        },
        {"l2": ppsci.metric.FunctionalMetric(pde_l2_rel_compute)},
        name="fg_L2_sup",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # update solver
    solver = ppsci.solver.Solver(
        solver.model,
        constraint_pde,
        OUTPUT_DIR,
        optimizer_pde,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=False,
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
    # maunally build constraint(s)
    train_dataloader_cfg_sol_f = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
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
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {"t": "t0", "x": "x0", "u_idn": "u0", "v_idn": "v0"},
        },
    }
    train_dataloader_cfg_sol_bc = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("x",),
            "alias_dict": {"t": "tb", "x": "xb"},
        },
    }

    sup_constraint_sol_f = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_sol_f,
        ppsci.loss.FunctionalLoss(pde_loss_compute),
        {
            "f_pde": lambda out: out["f_pde"],
            "g_pde": lambda out: out["g_pde"],
            "du_t": lambda out: ppsci.autodiff.jacobian(out["u_idn"], out["t"]),
            "dv_t": lambda out: ppsci.autodiff.jacobian(out["v_idn"], out["t"]),
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
        ppsci.loss.FunctionalLoss(boundary_loss_compute),
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

    # maunally build validator
    eval_dataloader_cfg_sol = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_SOL,
            "input_keys": ("t", "x"),
            "label_keys": ("u_idn", "v_idn"),
            "alias_dict": {
                "t": "t_star",
                "x": "x_star",
                "u_idn": "u_star",
                "v_idn": "v_star",
            },
        },
    }

    sup_validator_sol = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_sol,
        ppsci.loss.MSELoss("sum"),
        {key: (lambda out, k=key: out[k]) for key in ("u_idn", "v_idn")},
        {"l2": ppsci.metric.L2Rel()},
        name="uv_L2_sup",
    )
    validator_sol = {
        sup_validator_sol.name: sup_validator_sol,
    }

    # update solver
    solver = ppsci.solver.Solver(
        solver.model,
        constraint_sol,
        OUTPUT_DIR,
        optimizer_idn,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator_sol,
    )

    # train model
    solver.train()

    # Unused models can be deleted from model list if there is no enough cuda memory before eval
    del solver.model.model_list[-1]
    del solver.model.model_list[-1]
    # evaluate after finished training
    solver.eval()

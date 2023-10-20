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

"""
This module is heavily adapted from https://github.com/lululxvi/hpinn
"""

from os import path as osp

import functions as func_module
import hydra
import numpy as np
import paddle
import plotting as plot_module
from omegaconf import DictConfig

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import logger


def train_or_evaluate(cfg: DictConfig):
    # open FLAG for higher order differential operator
    paddle.framework.core.set_prim_eager_enabled(True)

    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # initialize models
    in_keys = ()
    for t in range(1, 7):
        in_keys += (f"x_cos_{t}", f"x_sin_{t}")
    in_keys += ("y", "y_cos_1", "y_sin_1")

    model_re = ppsci.arch.MLP(in_keys, **cfg.MODEL.re_net)
    model_im = ppsci.arch.MLP(in_keys, **cfg.MODEL.im_net)
    model_eps = ppsci.arch.MLP(in_keys, **cfg.MODEL.eps_net)

    # intialize params
    k = cfg.TRAIN_K
    func_module.train_mode = cfg.TRAIN_MODE
    loss_log_obj = []

    # register transform
    model_re.register_input_transform(func_module.transform_in)
    model_im.register_input_transform(func_module.transform_in)
    model_eps.register_input_transform(func_module.transform_in)

    model_re.register_output_transform(func_module.transform_out_real_part)
    model_im.register_output_transform(func_module.transform_out_imaginary_part)
    model_eps.register_output_transform(func_module.transform_out_epsilon)

    model_list = ppsci.arch.ModelList((model_re, model_im, model_eps))

    # initialize Adam optimizer
    optimizer_adam = ppsci.optimizer.Adam(cfg.TRAIN.learning_rate)(
        (model_re, model_im, model_eps)
    )

    # manually build constraint(s)
    label_keys = ("x", "y", "bound", "e_real", "e_imaginary", "epsilon")
    label_keys_derivative = (
        "de_re_x",
        "de_re_y",
        "de_re_xx",
        "de_re_yy",
        "de_im_x",
        "de_im_y",
        "de_im_xx",
        "de_im_yy",
    )
    output_expr = {
        "x": lambda out: out["x"],
        "y": lambda out: out["y"],
        "bound": lambda out: out["bound"],
        "e_real": lambda out: out["e_real"],
        "e_imaginary": lambda out: out["e_imaginary"],
        "epsilon": lambda out: out["epsilon"],
        "de_re_x": lambda out: jacobian(out["e_real"], out["x"]),
        "de_re_y": lambda out: jacobian(out["e_real"], out["y"]),
        "de_re_xx": lambda out: hessian(out["e_real"], out["x"]),
        "de_re_yy": lambda out: hessian(out["e_real"], out["y"]),
        "de_im_x": lambda out: jacobian(out["e_imaginary"], out["x"]),
        "de_im_y": lambda out: jacobian(out["e_imaginary"], out["y"]),
        "de_im_xx": lambda out: hessian(out["e_imaginary"], out["x"]),
        "de_im_yy": lambda out: hessian(out["e_imaginary"], out["y"]),
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableMatDataset",
                "file_path": cfg.DATASET_PATH,
                "input_keys": ("x", "y", "bound"),
                "label_keys": label_keys + label_keys_derivative,
                "alias_dict": {
                    "e_real": "x",
                    "e_imaginary": "x",
                    "epsilon": "x",
                    **{k: "x" for k in label_keys_derivative},
                },
            },
        },
        ppsci.loss.FunctionalLoss(func_module.pde_loss_fun),
        output_expr,
        name="sup_constraint_pde",
    )
    sup_constraint_obj = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "IterableMatDataset",
                "file_path": cfg.DATASET_PATH,
                "input_keys": ("x", "y", "bound"),
                "label_keys": label_keys,
                "alias_dict": {"e_real": "x", "e_imaginary": "x", "epsilon": "x"},
            },
        },
        ppsci.loss.FunctionalLoss(func_module.obj_loss_fun),
        {key: lambda out, k=key: out[k] for key in label_keys},
        name="sup_constraint_obj",
    )
    constraint = {
        sup_constraint_pde.name: sup_constraint_pde,
        sup_constraint_obj.name: sup_constraint_obj,
    }

    # manually build validator
    sup_validator_opt = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "IterableMatDataset",
                "file_path": cfg.DATASET_PATH_VALID,
                "input_keys": ("x", "y", "bound"),
                "label_keys": label_keys + label_keys_derivative,
                "alias_dict": {
                    "x": "x_opt",
                    "y": "y_opt",
                    "e_real": "x_opt",
                    "e_imaginary": "x_opt",
                    "epsilon": "x_opt",
                    **{k: "x_opt" for k in label_keys_derivative},
                },
            },
        },
        ppsci.loss.FunctionalLoss(func_module.eval_loss_fun),
        output_expr,
        {"mse": ppsci.metric.FunctionalMetric(func_module.eval_metric_fun)},
        name="opt_sup",
    )
    sup_validator_val = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "IterableMatDataset",
                "file_path": cfg.DATASET_PATH_VALID,
                "input_keys": ("x", "y", "bound"),
                "label_keys": label_keys + label_keys_derivative,
                "alias_dict": {
                    "x": "x_val",
                    "y": "y_val",
                    "e_real": "x_val",
                    "e_imaginary": "x_val",
                    "epsilon": "x_val",
                    **{k: "x_val" for k in label_keys_derivative},
                },
            },
        },
        ppsci.loss.FunctionalLoss(func_module.eval_loss_fun),
        output_expr,
        {"mse": ppsci.metric.FunctionalMetric(func_module.eval_metric_fun)},
        name="val_sup",
    )
    validator = {
        sup_validator_opt.name: sup_validator_opt,
        sup_validator_val.name: sup_validator_val,
    }

    if cfg.mode == "train":
        # initialize solver
        solver = ppsci.solver.Solver(
            model_list,
            constraint,
            cfg.output_dir,
            optimizer_adam,
            None,
            cfg.TRAIN.epochs,
            cfg.TRAIN.iters_per_epoch,
            eval_during_train=cfg.TRAIN.eval_during_train,
            validator=validator,
            checkpoint_path=cfg.TRAIN.checkpoint_path,
        )

        # train model
        solver.train()
        # evaluate after finished training
        solver.eval()

        # initialize LBFGS optimizer
        optimizer_lbfgs = ppsci.optimizer.LBFGS(max_iter=cfg.TRAIN.max_iter)(
            (model_re, model_im, model_eps)
        )

        # train: soft constraint, epoch=1 for lbfgs
        if cfg.TRAIN_MODE == "soft":
            solver = ppsci.solver.Solver(
                model_list,
                constraint,
                cfg.output_dir,
                optimizer_lbfgs,
                None,
                cfg.TRAIN.epochs_lbfgs,
                cfg.TRAIN.iters_per_epoch,
                eval_during_train=cfg.TRAIN.eval_during_train,
                validator=validator,
                checkpoint_path=cfg.TRAIN.checkpoint_path,
            )

            # train model
            solver.train()
            # evaluate after finished training
            solver.eval()

        # append objective loss for plot
        loss_log_obj.append(func_module.loss_obj)

        # penalty and augmented Lagrangian, difference between the two is updating of lambda
        if cfg.TRAIN_MODE != "soft":
            train_dict = ppsci.utils.reader.load_mat_file(
                cfg.DATASET_PATH, ("x", "y", "bound")
            )
            in_dict = {"x": train_dict["x"], "y": train_dict["y"]}
            expr_dict = output_expr.copy()
            expr_dict.pop("bound")

            func_module.init_lambda(in_dict, int(train_dict["bound"]))
            func_module.lambda_log.append(
                [
                    func_module.lambda_re.copy().squeeze(),
                    func_module.lambda_im.copy().squeeze(),
                ]
            )

            for i in range(1, k + 1):
                pred_dict = solver.predict(
                    in_dict,
                    expr_dict,
                    batch_size=np.shape(train_dict["x"])[0],
                    no_grad=False,
                )
                func_module.update_lambda(pred_dict, int(train_dict["bound"]))

                func_module.update_mu()
                logger.message(f"Iteration {i}: mu = {func_module.mu}\n")

                solver = ppsci.solver.Solver(
                    model_list,
                    constraint,
                    cfg.output_dir,
                    optimizer_lbfgs,
                    None,
                    cfg.TRAIN.epochs_lbfgs,
                    cfg.TRAIN.iters_per_epoch,
                    eval_during_train=cfg.TRAIN.eval_during_train,
                    validator=validator,
                    checkpoint_path=cfg.TRAIN.checkpoint_path,
                )

                # train model
                solver.train()
                # evaluate after finished training
                solver.eval()
                # append objective loss for plot
                loss_log_obj.append(func_module.loss_obj)

        ################# plotting ###################
        # log of loss
        loss_log = np.array(func_module.loss_log).reshape(-1, 3)

        plot_module.set_params(
            cfg.TRAIN_MODE, cfg.output_dir, cfg.DATASET_PATH, cfg.DATASET_PATH_VALID
        )
        plot_module.plot_6a(loss_log)
        if cfg.TRAIN_MODE != "soft":
            plot_module.prepare_data(solver, expr_dict)
            plot_module.plot_6b(loss_log_obj)
            plot_module.plot_6c7c(func_module.lambda_log)
            plot_module.plot_6d(func_module.lambda_log)
            plot_module.plot_6ef(func_module.lambda_log)
    elif cfg.mode == "eval":
        solver = ppsci.solver.Solver(
            model_list,
            constraint,
            cfg.output_dir,
            optimizer_adam,
            None,
            validator=validator,
            pretrained_model_path=cfg.EVAL.pretrained_model_path,
        )

        # train model
        solver.train()
        # evaluate after finished training
        solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="hpinns.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train" or cfg.mode == "eval":
        train_or_evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

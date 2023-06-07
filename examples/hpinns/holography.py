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
from functions import Funcs
from plotting import Plot

import ppsci
from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    # open FLAG for higher order differential operator
    paddle.fluid.core.set_prim_eager_enabled(True)

    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    DATASET_PATH = "./data/hpinns_train.mat"
    DATASET_PATH_VALID = "./data/hpinns_valid.mat"
    OUTPUT_DIR = "./output_hpinns/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize models
    in_keys = (
        "x_cos",
        "x_sin",
        "x_cos_2",
        "x_sin_2",
        "x_cos_3",
        "x_sin_3",
        "x_cos_4",
        "x_sin_4",
        "x_cos_5",
        "x_sin_5",
        "x_cos_6",
        "x_sin_6",
        "y",
        "y_cos",
        "y_sin",
    )
    model_re = ppsci.arch.MLP(in_keys, ("e_re",), 4, 48, "tanh")
    model_im = ppsci.arch.MLP(in_keys, ("e_im",), 4, 48, "tanh")
    model_eps = ppsci.arch.MLP(in_keys, ("eps",), 4, 48, "tanh")

    # intialize params
    f = Funcs()
    loss_log_obj = []
    train_mode = "aug_lag"  # "soft", "penalty", "aug_lag"
    k = 9

    # register transform
    model_re.register_input_transform(f.transform_in)
    model_im.register_input_transform(f.transform_in)
    model_eps.register_input_transform(f.transform_in)

    model_re.register_output_transform(f.transform_out_re)
    model_im.register_output_transform(f.transform_out_im)
    model_eps.register_output_transform(f.transform_out_eps)

    model_list = ppsci.arch.ModelList((model_re, model_im, model_eps))

    # set training hyper-parameters
    ITERS_PER_EPOCH = 1
    EPOCHS = 20000 if args.epochs is None else args.epochs
    MAX_ITER = 50000  # for LBFGS

    # initialize optimizer
    # Adam
    optimizer_adam = ppsci.optimizer.Adam(1e-3)((model_re, model_im, model_eps))
    # LBFGS
    optimizer_lbfgs = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)(
        (model_re, model_im, model_eps)
    )

    # maunally build constraint(s)
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

    train_dataloader_cfg_pde = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys + label_keys_derivative,
            "alias_dict": {
                "e_real": "x",
                "e_imaginary": "x",
                "epsilon": "x",
                "de_re_x": "x",
                "de_re_y": "x",
                "de_re_xx": "x",
                "de_re_yy": "x",
                "de_im_x": "x",
                "de_im_y": "x",
                "de_im_xx": "x",
                "de_im_yy": "x",
            },
        },
    }

    train_dataloader_cfg_obj = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys,
            "alias_dict": {"e_real": "x", "e_imaginary": "x", "epsilon": "x"},
        },
    }

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_pde,
        ppsci.loss.FunctionalLoss(f.pde_loss_fun),
        {
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
        },
        name="weighted_pde(&lag)_sup",
    )

    sup_constraint_obj = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg_obj,
        ppsci.loss.FunctionalLoss(f.obj_loss_fun),
        {key: lambda out, k=key: out[k] for key in label_keys},
        name="weighted_obj_sup",
    )

    constraint = {
        sup_constraint_pde.name: sup_constraint_pde,
        sup_constraint_obj.name: sup_constraint_obj,
    }

    # maunally build validator
    eval_dataloader_cfg_opt = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_VALID,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys + label_keys_derivative,
            "alias_dict": {
                "x": "x_opt",
                "y": "y_opt",
                "e_real": "x_opt",
                "e_imaginary": "x_opt",
                "epsilon": "x_opt",
                "de_re_x": "x_opt",
                "de_re_y": "x_opt",
                "de_re_xx": "x_opt",
                "de_re_yy": "x_opt",
                "de_im_x": "x_opt",
                "de_im_y": "x_opt",
                "de_im_xx": "x_opt",
                "de_im_yy": "x_opt",
            },
        },
    }

    eval_dataloader_cfg_val = {
        "dataset": {
            "name": "IterableMatDataset",
            "file_path": DATASET_PATH_VALID,
            "input_keys": ("x", "y", "bound"),
            "label_keys": label_keys + label_keys_derivative,
            "alias_dict": {
                "x": "x_val",
                "y": "y_val",
                "e_real": "x_val",
                "e_imaginary": "x_val",
                "epsilon": "x_val",
                "de_re_x": "x_val",
                "de_re_y": "x_val",
                "de_re_xx": "x_val",
                "de_re_yy": "x_val",
                "de_im_x": "x_val",
                "de_im_y": "x_val",
                "de_im_xx": "x_val",
                "de_im_yy": "x_val",
            },
        },
    }

    sup_validator_opt = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_opt,
        ppsci.loss.FunctionalLoss(f.eval_loss_fun),
        {
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
        },
        {"func": ppsci.metric.FunctionalMetric(f.eval_metric_fun)},
        name="opt_sup",
    )

    sup_validator_val = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg_val,
        ppsci.loss.FunctionalLoss(f.eval_loss_fun),
        {
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
        },
        {"func": ppsci.metric.FunctionalMetric(f.eval_metric_fun)},
        name="val_sup",
    )

    validator = {
        sup_validator_opt.name: sup_validator_opt,
        sup_validator_val.name: sup_validator_val,
    }

    # initialize solver
    # train: base, epoch=EPOCHS
    solver = ppsci.solver.Solver(
        model_list,
        constraint,
        OUTPUT_DIR,
        optimizer_adam,
        None,
        EPOCHS,
        ITERS_PER_EPOCH,
        eval_during_train=False,
        validator=validator,
    )

    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()

    # train: soft constraint, epoch=1 for lbfgs
    if train_mode is "soft":
        solver = ppsci.solver.Solver(
            solver.model,
            constraint,
            OUTPUT_DIR,
            optimizer_lbfgs,
            None,
            1,
            ITERS_PER_EPOCH,
            eval_during_train=False,
            validator=validator,
        )

        # train model
        solver.train()
        # evaluate after finished training
        solver.eval()

    # append objective loss for plot
    loss_log_obj.append(f.loss_opt)

    # penalty and augmented Lagrangian, difference between the two is updating of lambda
    if train_mode is not "soft":
        # only needed by augmented Lagrangian
        if train_mode is "aug_lag":
            train_dict = f.load_data_from_mat(DATASET_PATH)
            in_dict = {"x": train_dict["x"], "y": train_dict["y"]}
            expr_dict = {
                "x": lambda out: out["x"],
                "y": lambda out: out["y"],
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
            f.init_lambda(in_dict, int(train_dict["bound"]))
            f.lambda_log.append(
                [f.lambda_re.copy().squeeze(), f.lambda_im.copy().squeeze()]
            )

        for i in range(1, k + 1):
            # only needed by augmented Lagrangian
            if train_mode is "aug_lag":
                pred_dict = solver.predict(
                    in_dict,
                    expr_dict,
                    batch_size=np.shape(train_dict["x"])[0],
                    no_grad=False,
                )
                f.update_lambda(pred_dict, int(train_dict["bound"]))

            f.update_mu()
            print(f"Iteration {i}: mu = {f.mu}\n")

            solver = ppsci.solver.Solver(
                solver.model,
                constraint,
                OUTPUT_DIR,
                optimizer_lbfgs,
                None,
                1,
                ITERS_PER_EPOCH,
                eval_during_train=False,
                validator=validator,
            )

            # train model
            solver.train()
            # evaluate after finished training
            solver.eval()
            # append objective loss for plot
            loss_log_obj.append(f.loss_opt)

    ################# plotting ###################
    # log of loss
    loss_log = np.array(f.loss_log).reshape(-1, 3)

    p = Plot(train_mode, OUTPUT_DIR, DATASET_PATH, DATASET_PATH_VALID, f)
    p.prepare_data(solver)
    p.plot_6a(loss_log)
    if train_mode is not "soft":
        p.plot_6b(loss_log_obj)
        p.plot_6c7c(f.lambda_log)
        p.plot_6d(f.lambda_log)
        p.plot_6ef(f.lambda_log)

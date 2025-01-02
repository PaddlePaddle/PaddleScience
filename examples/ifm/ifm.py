# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import hydra
import numpy as np
from ednn_utils import Meter
from omegaconf import DictConfig
from paddle.nn import BCEWithLogitsLoss
from paddle.nn import MSELoss

import ppsci
from ppsci.utils import logger


def get_train_loss_func(reg, pos_weights=None):  #:paddle.Tensor=None):
    def train_loss_func(output_dict, label_dict, weight_dict):
        if reg:
            loss_func = MSELoss(reduction="none")
        else:
            loss_func = BCEWithLogitsLoss(reduction="none", pos_weight=pos_weights)
        return {
            "pred": (
                loss_func(output_dict["pred"], label_dict["y"])
                * (label_dict["mask"] != 0).astype("float32")
            ).mean()
        }

    return train_loss_func


def get_val_loss_func(reg, metric):
    def val_loss_func(output_dict, label_dict):
        eval_metric = Meter()
        eval_metric.update(output_dict["pred"], label_dict["y"], label_dict["mask"])

        if reg:
            rmse_score = np.mean(eval_metric.compute_metric(metric))
            mae_score = np.mean(eval_metric.compute_metric("mae"))
            r2_score = np.mean(eval_metric.compute_metric("r2"))
            return {"rmse": rmse_score, "mae": mae_score, "r2": r2_score}
        else:
            roc_score = np.mean(eval_metric.compute_metric(metric))
            prc_score = np.mean(eval_metric.compute_metric("prc_auc"))
            return {"roc_auc": roc_score, "prc_auc": prc_score}

    return val_loss_func


def train(cfg: DictConfig):
    if cfg.data_label in ["esol", "freesolv", "lipop"]:
        # task_type = "reg"
        reg = True
        metric = "rmse"
    else:
        # task_type = "cla"
        reg = False
        metric = "roc_auc"

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "train",
            "data_label": cfg.data_label,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        name="Sup",
    )

    # parmas from dataset
    inputs = sup_constraint.data_loader.dataset.data_tr_x.shape[1]
    tasks = sup_constraint.data_loader.dataset.task_dict[cfg.data_label]
    iters_per_epoch = len(sup_constraint.data_loader)
    logger.info(f"inputs is: {inputs}, iters_per_epoch: {iters_per_epoch}")
    if not reg:
        pos_weights = sup_constraint.data_loader.dataset.pos_weights
        sup_constraint.loss = ppsci.loss.FunctionalLoss(
            get_train_loss_func(reg, pos_weights)
        )

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    hyper_paras = cfg.HYPER_OPT[cfg.data_label]

    hidden_units = [
        hyper_paras["hidden_unit1"],
        hyper_paras["hidden_unit2"],
        hyper_paras["hidden_unit3"],
    ]
    # set model
    model = ppsci.arch.IFMMLP(
        # **cfg.MODEL,
        input_keys=("x",),
        output_keys=("pred",),
        hidden_units=hidden_units,
        embed_name=cfg.MODEL.embed_name,
        inputs=inputs,
        outputs=len(tasks),
        d_out=hyper_paras["d_out"],
        sigma=hyper_paras["sigma"],
        dp_ratio=hyper_paras["dropout"],
        reg=reg,
        first_omega_0=hyper_paras["omega0"],
        hidden_omega_0=hyper_paras["omega1"],
    )

    # set optimizer
    optimizer = ppsci.optimizer.Adam(
        learning_rate=cfg.TRAIN.learning_rate, weight_decay=hyper_paras["l2"]
    )(model)

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "val",
            "data_label": cfg.data_label,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        output_expr={"pred": lambda out: out["pred"]},
        metric={
            "MyMeter": ppsci.metric.FunctionalMetric(get_val_loss_func(reg, metric))
        },
        name="MyMeter_validator",
    )
    if not reg:
        pos_weights = rmse_validator.data_loader.dataset.pos_weights
        rmse_validator.loss = ppsci.loss.FunctionalLoss(
            get_train_loss_func(reg, pos_weights)
        )

    validator = {rmse_validator.name: rmse_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.HYPER_OPT[cfg.data_label].epoch,  # cfg.TRAIN.epochs,
        iters_per_epoch,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        validator=validator,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # train model
    solver.train()


def evaluate(cfg: DictConfig):
    if cfg.data_label in ["esol", "freesolv", "lipop"]:
        # task_type = "reg"
        reg = True
        metric = "rmse"
    else:
        # task_type = "cla"
        reg = False
        metric = "roc_auc"

    # set dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "train",
            "data_label": cfg.data_label,
        },
        "batch_size": 128,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        eval_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        name="Sup",
    )

    inputs = sup_constraint.data_loader.dataset.data_tr_x.shape[1]
    tasks = sup_constraint.data_loader.dataset.task_dict[cfg.data_label]

    hyper_paras = cfg.HYPER_OPT[cfg.data_label]
    hidden_units = [
        hyper_paras["hidden_unit1"],
        hyper_paras["hidden_unit2"],
        hyper_paras["hidden_unit3"],
    ]
    print(f"hyper_params = {hyper_paras}")

    # set model
    model = ppsci.arch.IFMMLP(
        # **cfg.MODEL,
        input_keys=("x",),
        output_keys=("pred",),
        hidden_units=hidden_units,
        embed_name=cfg.MODEL.embed_name,
        inputs=inputs,
        outputs=len(tasks),
        d_out=hyper_paras["d_out"],
        sigma=hyper_paras["sigma"],
        dp_ratio=hyper_paras["dropout"],
        reg=reg,
        first_omega_0=hyper_paras["omega0"],
        hidden_omega_0=hyper_paras["omega1"],
    )

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "test",
            "data_label": cfg.data_label,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        output_expr={"pred": lambda out: out["pred"]},
        metric={
            "MyMeter": ppsci.metric.FunctionalMetric(get_val_loss_func(reg, metric))
        },
        name="MyMeter_validator",
    )
    if not reg:
        pos_weights = rmse_validator.data_loader.dataset.pos_weights
        rmse_validator.loss = ppsci.loss.FunctionalLoss(
            get_train_loss_func(reg, pos_weights)
        )

    validator = {rmse_validator.name: rmse_validator}

    if cfg.EVAL.pretrained_model_path:
        pretrained_model_path = cfg.EVAL.pretrained_model_path
    else:
        t_epoch = cfg.HYPER_OPT[cfg.data_label].epoch
        load_epoch = t_epoch - t_epoch % cfg.TRAIN.save_freq
        pretrained_model_path = os.path.join(
            cfg.output_dir, "checkpoints", "epoch_" + str(load_epoch) + ".pdparams"
        )

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        validator=validator,
        pretrained_model_path=pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="ifm.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

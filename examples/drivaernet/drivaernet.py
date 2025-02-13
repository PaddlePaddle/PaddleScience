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

import warnings
from functools import partial

import hydra
import paddle
from omegaconf import DictConfig

import ppsci


def train(cfg: DictConfig):
    # set model
    model = ppsci.arch.RegDGCNN(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL,
    )

    train_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.TRAIN.train_ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.TRAIN.num_points,
            "train_fractions": cfg.TRAIN.train_fractions,
            "mode": cfg.mode,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernet_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="DrivAerNet_constraint",
    )

    constraint = {drivaernet_constraint.name: drivaernet_constraint}

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.TRAIN.eval_ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.TRAIN.num_points,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernet_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="DrivAerNet_valid",
    )

    validator = {drivaernet_valid.name: drivaernet_valid}

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ReduceOnPlateau(
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=(
            cfg.TRAIN.iters_per_epoch
            * cfg.TRAIN.train_fractions
            // (paddle.distributed.get_world_size() * cfg.TRAIN.batch_size)
            + 1
        ),
        learning_rate=cfg.optimizer.lr,
        mode=cfg.TRAIN.scheduler.mode,
        patience=cfg.TRAIN.scheduler.patience,
        factor=cfg.TRAIN.scheduler.factor,
        verbose=cfg.TRAIN.scheduler.verbose,
    )()

    optimizer = (
        ppsci.optimizer.Adam(lr_scheduler, weight_decay=cfg.optimizer.weight_decay)(
            model
        )
        if cfg.optimizer.optimizer == "adam"
        else ppsci.optimizer.SGD(lr_scheduler, weight_decay=cfg.optimizer.weight_decay)(
            model
        )
    )

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        iters_per_epoch=(
            cfg.TRAIN.iters_per_epoch
            * cfg.TRAIN.train_fractions
            // (paddle.distributed.get_world_size() * cfg.TRAIN.batch_size)
            + 1
        ),
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=cfg.TRAIN.epochs,
        validator=validator,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    lr_scheduler.step = partial(lr_scheduler.step, metrics=solver.cur_metric)
    solver.lr_scheduler = lr_scheduler

    # train model
    solver.train()

    solver.eval()


def evaluate(cfg: DictConfig):
    # set model
    model = ppsci.arch.RegDGCNN(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL,
    )

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.EVAL.ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.EVAL.num_points,
            "mode": cfg.mode,
        },
        "batch_size": cfg.EVAL.batch_size,
        "num_workers": cfg.EVAL.num_workers,
    }

    drivaernet_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={
            "MSE": ppsci.metric.MSE(),
            "MAE": ppsci.metric.MAE(),
            "Max AE": ppsci.metric.MaxAE(),
            "R²": ppsci.metric.R2Score(),
        },
        name="DrivAerNet_valid",
    )

    validator = {drivaernet_valid.name: drivaernet_valid}

    solver = ppsci.solver.Solver(
        model=model,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="drivaernet.yaml")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore")
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

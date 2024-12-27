# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import path as osp

import hydra
import paddle
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger


def train(cfg: DictConfig):
    # set seed
    ppsci.utils.misc.set_random_seed(cfg.TRAIN.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

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
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
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
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
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
    # set optimizer
    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(mode=cfg.TRAIN.scheduler.mode,
                                                       patience=cfg.TRAIN.scheduler.patience,
                                                       factor=cfg.TRAIN.scheduler.factor,
                                                       verbose=cfg.TRAIN.scheduler.verbose,
                                                       learning_rate=cfg.ARGS.lr)

    optimizer = ppsci.optimizer.Adam(lr_scheduler, weight_decay=cfg.ARGS.weight_decay)(
        model) if cfg.ARGS.optimizer == 'adam' else ppsci.optimizer.SGD(
        lr_scheduler, weight_decay=cfg.ARGS.weight_decay)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=cfg.TRAIN.epochs,
        validator=validator,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # train model
    solver.train()

    solver.eval()


def evaluate(cfg: DictConfig):
    # set seed
    ppsci.utils.misc.set_random_seed(cfg.TRAIN.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

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
            "num_points": cfg.TRAIN.num_points,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernet_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="DrivAerNet_valid",
    )

    validator = {drivaernet_valid.name: drivaernet_valid}

    solver = ppsci.solver.Solver(
        model=model,
        output_dir=cfg.output_dir,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="DriveAerNet.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

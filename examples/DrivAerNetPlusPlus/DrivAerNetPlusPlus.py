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
    model = ppsci.arch.RegPointNet(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL,
    )

    train_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetPlusPlusDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.TRAIN.train_ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.MODEL.num_points,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernetplusplus_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        name="DrivAerNetplusplus_constraint",
    )

    constraint = {drivaernetplusplus_constraint.name: drivaernetplusplus_constraint}

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetPlusPlusDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.TRAIN.eval_ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.MODEL.num_points,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernetplusplus_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="DrivAerNetplusplus_valid",
    )

    validator = {drivaernetplusplus_valid.name: drivaernetplusplus_valid}

    # set optimizer
    optimizer = (
        paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=cfg.ARGS.lr,
            weight_decay=cfg.ARGS.weight_decay,
        )
        if cfg.ARGS.optimizer == "adam"
        else paddle.optimizer.SGD(
            parameters=model.parameters(),
            learning_rate=cfg.ARGS.lr,
            weight_decay=cfg.ARGS.weight_decay,
        )
    )

    tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(
        mode=cfg.TRAIN.scheduler.mode,
        patience=cfg.TRAIN.scheduler.patience,
        factor=cfg.TRAIN.scheduler.factor,
        verbose=cfg.TRAIN.scheduler.verbose,
        learning_rate=optimizer.get_lr(),
    )
    optimizer.set_lr_scheduler(tmp_lr)
    scheduler = tmp_lr

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        lr_scheduler=scheduler,
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
    model = ppsci.arch.RegPointNet(
        input_keys=cfg.MODEL.input_keys,
        label_keys=cfg.MODEL.output_keys,
        weight_keys=cfg.MODEL.weight_keys,
        args=cfg.MODEL,
    )

    valid_dataloader_cfg = {
        "dataset": {
            "name": "DrivAerNetPlusPlusDataset",
            "root_dir": cfg.ARGS.dataset_path,
            "input_keys": ("vertices",),
            "label_keys": ("cd_value",),
            "weight_keys": ("weight_keys",),
            "subset_dir": cfg.ARGS.subset_dir,
            "ids_file": cfg.EVAL.ids_file,
            "csv_file": cfg.ARGS.aero_coeff,
            "num_points": cfg.MODEL.num_points,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": cfg.TRAIN.num_workers,
    }

    drivaernetplusplus_valid = ppsci.validate.SupervisedValidator(
        valid_dataloader_cfg,
        loss=ppsci.loss.MSELoss("mean"),
        metric={"MSE": ppsci.metric.MSE()},
        name="DrivAerNetPlusPlus_valid",
    )

    validator = {drivaernetplusplus_valid.name: drivaernetplusplus_valid}

    solver = ppsci.solver.Solver(
        model=model,
        output_dir=cfg.output_dir,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(
    version_base=None, config_path="./conf", config_name="DriveAerNetPlusPlus.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import hydra
import utils as utils
from omegaconf import DictConfig

import ppsci


def train(cfg: DictConfig):
    # set train dataloader config
    if not cfg.USE_SAMPLED_DATA:
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5SQDataset",
                "file_path": cfg.TRAIN_FILE_PATH,
                "input_keys": cfg.MODEL.input_keys,
                "label_keys": cfg.MODEL.output_keys,
                "size": (cfg.IMG_H, cfg.IMG_W),
            },
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": 4,
        }
    else:
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5SampledDataset",
                "file_path": cfg.TRAIN_FILE_PATH,
                "input_keys": cfg.MODEL.input_keys,
                "label_keys": cfg.MODEL.output_keys,
            },
            "sampler": {
                "name": "DistributedBatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": 4,
        }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.MSELoss(),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5SQDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "training": False,
            "size": (cfg.IMG_H, cfg.IMG_W),
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        metric={
            "MAE": ppsci.metric.MAE(keep_batch=True),
            "MSE": ppsci.metric.MSE(keep_batch=True),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.Preformer(**cfg.MODEL)

    # init optimizer and lr scheduler
    lr_scheduler_cfg = dict(cfg.TRAIN.lr_scheduler)
    lr_scheduler_cfg.update({"iters_per_epoch": ITERS_PER_EPOCH})
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(**lr_scheduler_cfg)()

    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        output_dir=cfg.output_dir,
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        iters_per_epoch=ITERS_PER_EPOCH,
        log_freq=cfg.log_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        device=cfg.device,
        validator=validator,
        compute_metric_by_batch=True,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5SQDataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.input_keys,
            "label_keys": cfg.MODEL.output_keys,
            "training": False,
            "size": (cfg.IMG_H, cfg.IMG_W),
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss(),
        metric={
            "MAE": ppsci.metric.MAE(keep_batch=True),
            "MSE": ppsci.metric.MSE(keep_batch=True),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.Preformer(**cfg.MODEL)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="preformer.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

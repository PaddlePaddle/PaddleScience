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
import paddle.distributed as dist
from omegaconf import DictConfig

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import logger


def get_data_stat(cfg: DictConfig):
    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.DATA_TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W, cfg.VARS_CHANNEL
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )
    return data_mean, data_std, data_time_mean_normalize


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")
    # Initialize distributed environment
    dist.init_parallel_env()

    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.DATA_TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W, cfg.VARS_CHANNEL
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )
    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (cfg.IMG_H, cfg.IMG_W)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]

    # set train dataloader config
    if not cfg.USE_SAMPLED_DATA:
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5Dataset",
                "file_path": cfg.TRAIN_FILE_PATH,
                "input_keys": cfg.input_keys,
                "label_keys": cfg.output_keys,
                "vars_channel": cfg.VARS_CHANNEL,
                "transforms": transforms,
            },
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": True,
            },
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": cfg.TRAIN.num_workers,
        }
    else:
        NUM_GPUS_PER_NODE = 8
        train_dataloader_cfg = {
            "dataset": {
                "name": "ERA5SampledDataset",
                "file_path": cfg.TRAIN_FILE_PATH,
                "input_keys": cfg.input_keys,
                "label_keys": cfg.output_keys,
            },
            "sampler": {
                "name": "DistributedBatchSampler",
                "drop_last": True,
                "shuffle": True,
                "num_replicas": NUM_GPUS_PER_NODE,
                "rank": dist.get_rank() % NUM_GPUS_PER_NODE,
            },
            "batch_size": cfg.TRAIN.batch_size,
            "num_workers": cfg.TRAIN.num_workers,
        }
    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        name="Sup",
    )
    constraint = {sup_constraint.name: sup_constraint}

    # set iters_per_epoch by dataloader length
    ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.input_keys,
            "label_keys": cfg.output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "transforms": transforms,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={
            "MAE": ppsci.metric.MAE(keep_batch=True),
            "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
                num_lat=cfg.IMG_H,
                std=data_std,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
            "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
                num_lat=cfg.IMG_H,
                mean=data_time_mean_normalize,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.AFNONet(**cfg.MODEL.afno)

    # init optimizer and lr scheduler
    lr_scheduler_cfg = dict(cfg.TRAIN.lr_scheduler)
    lr_scheduler_cfg.update({"iters_per_epoch": ITERS_PER_EPOCH})
    lr_scheduler = ppsci.optimizer.lr_scheduler.Cosine(**lr_scheduler_cfg)()

    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        lr_scheduler,
        cfg.TRAIN.epochs,
        ITERS_PER_EPOCH,
        eval_during_train=True,
        seed=cfg.seed,
        validator=validator,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # train model
    solver.train()
    # evaluate after finished training
    solver.eval()


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.DATA_TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W, cfg.VARS_CHANNEL
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )
    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (cfg.IMG_H, cfg.IMG_W)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.input_keys,
            "label_keys": cfg.output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "transforms": transforms,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric={
            "MAE": ppsci.metric.MAE(keep_batch=True),
            "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
                num_lat=cfg.IMG_H,
                std=data_std,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
            "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
                num_lat=cfg.IMG_H,
                mean=data_time_mean_normalize,
                keep_batch=True,
                variable_dict={"u10": 0, "v10": 1},
            ),
        },
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model = ppsci.arch.AFNONet(**cfg.MODEL.afno)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        seed=cfg.seed,
        validator=validator,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    # evaluate
    solver.eval()


@hydra.main(
    version_base=None, config_path="./conf", config_name="fourcastnet_pretrain.yaml"
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

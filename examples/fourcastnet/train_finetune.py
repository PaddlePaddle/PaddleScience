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

import functools
from os import path as osp
from typing import Tuple

import h5py
import hydra
import numpy as np
import paddle
import paddle.distributed as dist
from omegaconf import DictConfig

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import logger


def get_vis_data(
    file_path: str,
    date_strings: Tuple[str, ...],
    num_timestamps: int,
    vars_channel: Tuple[int, ...],
    img_h: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
):
    _file = h5py.File(file_path, "r")["fields"]
    data = []
    for date_str in date_strings:
        hours_since_jan_01_epoch = fourcast_utils.date_to_hours(date_str)
        ic = int(hours_since_jan_01_epoch / 6)
        data.append(_file[ic : ic + num_timestamps + 1, vars_channel, 0:img_h])
    data = np.asarray(data)

    vis_data = {"input": (data[:, 0] - data_mean) / data_std}
    for t in range(num_timestamps):
        hour = (t + 1) * 6
        data_t = data[:, t + 1]
        wind_data = []
        for i in range(data_t.shape[0]):
            wind_data.append((data_t[i][0] ** 2 + data_t[i][1] ** 2) ** 0.5)
        vis_data[f"target_{hour}h"] = np.asarray(wind_data)
    return vis_data


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(1024)
    # Initialize distributed environment
    dist.init_parallel_env()

    # initialize logger
    logger.init_logger("ppsci", f"{cfg.output_dir}/train.log", "info")

    # set training hyper-parameters
    output_keys = tuple(f"output_{i}" for i in range(cfg.TRAIN.num_timestamps))

    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.DATA_TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W, cfg.VARS_CHANNEL
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )

    # set transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (cfg.IMG_H, cfg.IMG_W)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]
    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.TRAIN_FILE_PATH,
            "input_keys": cfg.input_keys,
            "label_keys": output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "num_label_timestamps": cfg.TRAIN.num_timestamps,
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
            "label_keys": output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "transforms": transforms,
            "num_label_timestamps": cfg.TRAIN.num_timestamps,
            "training": False,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set metric
    metric = {
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
    }

    # set validator
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric=metric,
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set model
    model_cfg = dict(cfg.MODEL.afno)
    model_cfg.update(
        {"output_keys": output_keys, "num_timestamps": cfg.TRAIN.num_timestamps}
    )

    model = ppsci.arch.AFNONet(**model_cfg)

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
        validator=validator,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
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

    # set testing hyper-parameters
    output_keys = tuple(f"output_{i}" for i in range(cfg.EVAL.num_timestamps))

    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.DATA_TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W, cfg.VARS_CHANNEL
    )
    data_time_mean_normalize = np.expand_dims(
        (data_time_mean[0] - data_mean) / data_std, 0
    )

    # set transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (cfg.IMG_H, cfg.IMG_W)}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]

    # set model
    model_cfg = dict(cfg.MODEL.afno)
    model_cfg.update(
        {"output_keys": output_keys, "num_timestamps": cfg.EVAL.num_timestamps}
    )
    model = ppsci.arch.AFNONet(**model_cfg)

    # set eval dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.TEST_FILE_PATH,
            "input_keys": cfg.input_keys,
            "label_keys": output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "transforms": transforms,
            "num_label_timestamps": cfg.EVAL.num_timestamps,
            "training": False,
            "stride": 8,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # set metirc
    metric = {
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
    }

    # set validator for testing
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.L2RelLoss(),
        metric=metric,
        name="Sup_Validator",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer data
    DATE_STRINGS = ("2018-09-08 00:00:00",)
    vis_data = get_vis_data(
        cfg.TEST_FILE_PATH,
        DATE_STRINGS,
        cfg.EVAL.num_timestamps,
        cfg.VARS_CHANNEL,
        cfg.IMG_H,
        data_mean,
        data_std,
    )

    def output_wind_func(d, var_name, data_mean, data_std):
        output = (d[var_name] * data_std) + data_mean
        wind_data = []
        for i in range(output.shape[0]):
            wind_data.append((output[i][0] ** 2 + output[i][1] ** 2) ** 0.5)
        return paddle.to_tensor(wind_data, paddle.get_default_dtype())

    vis_output_expr = {}
    for i in range(cfg.EVAL.num_timestamps):
        hour = (i + 1) * 6
        vis_output_expr[f"output_{hour}h"] = functools.partial(
            output_wind_func,
            var_name=f"output_{i}",
            data_mean=paddle.to_tensor(data_mean, paddle.get_default_dtype()),
            data_std=paddle.to_tensor(data_std, paddle.get_default_dtype()),
        )
        vis_output_expr[f"target_{hour}h"] = lambda d, hour=hour: d[f"target_{hour}h"]
    # set visualizer
    visualizer = {
        "visualize_wind": ppsci.visualize.VisualizerWeather(
            vis_data,
            vis_output_expr,
            xticks=np.linspace(0, 1439, 13),
            xticklabels=[str(i) for i in range(360, -1, -30)],
            yticks=np.linspace(0, 719, 7),
            yticklabels=[str(i) for i in range(90, -91, -30)],
            vmin=0,
            vmax=25,
            colorbar_label="m\s",
            batch_size=cfg.EVAL.batch_size,
            num_timestamps=cfg.EVAL.num_timestamps,
            prefix="wind",
        )
    }

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        validator=validator,
        visualizer=visualizer,
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
        compute_metric_by_batch=cfg.EVAL.compute_metric_by_batch,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )
    solver.eval()
    # visualize prediction from pretrained_model_path
    solver.visualize()


@hydra.main(
    version_base=None, config_path="./conf", config_name="fourcastnet_finetune.yaml"
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

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
import os.path as osp
from typing import Tuple

import h5py
import hydra
import numpy as np
import paddle
from omegaconf import DictConfig

import examples.fourcastnet.utils as fourcast_utils
import ppsci
from ppsci.utils import logger


def get_vis_data(
    wind_file_path: str,
    file_path: str,
    date_strings: Tuple[str, ...],
    num_timestamps: int,
    vars_channel: Tuple[int, ...],
    img_h: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
):
    __wind_file = h5py.File(wind_file_path, "r")["fields"]
    _file = h5py.File(file_path, "r")["tp"]
    wind_data = []
    data = []
    for date_str in date_strings:
        hours_since_jan_01_epoch = fourcast_utils.date_to_hours(date_str)
        ic = int(hours_since_jan_01_epoch / 6)
        wind_data.append(__wind_file[ic, vars_channel, 0:img_h])
        data.append(_file[ic + 1 : ic + num_timestamps + 1, 0:img_h])
    wind_data = np.asarray(wind_data)
    data = np.asarray(data)

    vis_data = {"input": (wind_data - data_mean) / data_std}
    for t in range(num_timestamps):
        hour = (t + 1) * 6
        data_t = data[:, t]
        vis_data[f"target_{hour}h"] = np.asarray(data_t)
    return vis_data


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", f"{cfg.output_dir}/train.log", "info")

    wind_data_mean, wind_data_std = fourcast_utils.get_mean_std(
        cfg.WIND_MEAN_PATH, cfg.WIND_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W
    )

    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (cfg.IMG_H, cfg.IMG_W)}},
        {
            "Normalize": {
                "mean": wind_data_mean,
                "std": wind_data_std,
                "apply_keys": ("input",),
            }
        },
        {"Log1p": {"scale": 1e-5, "apply_keys": ("label",)}},
    ]

    # set train dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.WIND_TRAIN_FILE_PATH,
            "input_keys": cfg.MODEL.precip.input_keys,
            "label_keys": cfg.MODEL.precip.output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "precip_file_path": cfg.TRAIN_FILE_PATH,
            "transforms": transforms,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "batch_size": cfg.TRAIN.batch_size,
        "num_workers": 8,
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
            "file_path": cfg.WIND_VALID_FILE_PATH,
            "input_keys": cfg.MODEL.precip.input_keys,
            "label_keys": cfg.MODEL.precip.output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "precip_file_path": cfg.VALID_FILE_PATH,
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

    # set metric
    metric = {
        "MAE": ppsci.metric.MAE(keep_batch=True),
        "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
            num_lat=cfg.IMG_H, keep_batch=True, unlog=True
        ),
        "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
            num_lat=cfg.IMG_H, mean=data_time_mean, keep_batch=True, unlog=True
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
    wind_model = ppsci.arch.AFNONet(**cfg.MODEL.afno)
    ppsci.utils.save_load.load_pretrain(wind_model, path=cfg.WIND_MODEL_PATH)
    model_cfg = dict(cfg.MODEL.precip)
    model_cfg.update({"wind_model": wind_model})
    model = ppsci.arch.PrecipNet(**model_cfg)

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

    # set model for testing
    wind_model = ppsci.arch.AFNONet(**cfg.MODEL.afno)
    ppsci.utils.save_load.load_pretrain(wind_model, path=cfg.WIND_MODEL_PATH)
    model_cfg = dict(cfg.MODEL.precip)
    model_cfg.update(
        {
            "output_keys": output_keys,
            "num_timestamps": cfg.EVAL.num_timestamps,
            "wind_model": wind_model,
        }
    )
    model = ppsci.arch.PrecipNet(**model_cfg)

    wind_data_mean, wind_data_std = fourcast_utils.get_mean_std(
        cfg.WIND_MEAN_PATH, cfg.WIND_STD_PATH, cfg.VARS_CHANNEL
    )
    data_time_mean = fourcast_utils.get_time_mean(
        cfg.TIME_MEAN_PATH, cfg.IMG_H, cfg.IMG_W
    )

    # set train transforms
    transforms = [
        {"SqueezeData": {}},
        {"CropData": {"xmin": (0, 0), "xmax": (cfg.IMG_H, cfg.IMG_W)}},
        {
            "Normalize": {
                "mean": wind_data_mean,
                "std": wind_data_std,
                "apply_keys": ("input",),
            }
        },
        {"Log1p": {"scale": 1e-5, "apply_keys": ("label",)}},
    ]

    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.WIND_TEST_FILE_PATH,
            "input_keys": cfg.MODEL.precip.input_keys,
            "label_keys": output_keys,
            "vars_channel": cfg.VARS_CHANNEL,
            "precip_file_path": cfg.TEST_FILE_PATH,
            "num_label_timestamps": cfg.EVAL.num_timestamps,
            "stride": 8,
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
    # set metirc
    metric = {
        "MAE": ppsci.metric.MAE(keep_batch=True),
        "LatitudeWeightedRMSE": ppsci.metric.LatitudeWeightedRMSE(
            num_lat=cfg.IMG_H, keep_batch=True, unlog=True
        ),
        "LatitudeWeightedACC": ppsci.metric.LatitudeWeightedACC(
            num_lat=cfg.IMG_H, mean=data_time_mean, keep_batch=True, unlog=True
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

    # set set visualizer data
    DATE_STRINGS = ("2018-04-04 00:00:00",)
    vis_data = get_vis_data(
        cfg.WIND_TEST_FILE_PATH,
        cfg.TEST_FILE_PATH,
        DATE_STRINGS,
        cfg.EVAL.num_timestamps,
        cfg.VARS_CHANNEL,
        cfg.IMG_H,
        wind_data_mean,
        wind_data_std,
    )

    def output_precip_func(d, var_name):
        output = 1e-2 * paddle.expm1(d[var_name][0])
        return output

    visu_output_expr = {}
    for i in range(cfg.EVAL.num_timestamps):
        hour = (i + 1) * 6
        visu_output_expr[f"output_{hour}h"] = functools.partial(
            output_precip_func,
            var_name=f"output_{i}",
        )
        visu_output_expr[f"target_{hour}h"] = (
            lambda d, hour=hour: d[f"target_{hour}h"] * 1000
        )
    # set visualizer
    visualizer = {
        "visualize_precip": ppsci.visualize.VisualizerWeather(
            vis_data,
            visu_output_expr,
            xticks=np.linspace(0, 1439, 13),
            xticklabels=[str(i) for i in range(360, -1, -30)],
            yticks=np.linspace(0, 719, 7),
            yticklabels=[str(i) for i in range(90, -91, -30)],
            vmin=0.001,
            vmax=130,
            colorbar_label="mm",
            log_norm=True,
            batch_size=cfg.EVAL.batch_size,
            num_timestamps=cfg.EVAL.num_timestamps,
            prefix="precip",
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
    # visualize prediction
    solver.visualize()


@hydra.main(
    version_base=None, config_path="./conf", config_name="fourcastnet_precip.yaml"
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

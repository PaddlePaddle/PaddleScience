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
import os
from os import path as osp
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
    ppsci.utils.set_random_seed(cfg.seed)

    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "train.log"), "info")

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
            "input_keys": cfg.MODEL.afno.input_keys,
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
            "file_path": cfg.VALID_FILE_PATH,
            "input_keys": cfg.MODEL.afno.input_keys,
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
            "input_keys": cfg.MODEL.afno.input_keys,
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


def export(cfg: DictConfig):
    # set model
    model = ppsci.arch.AFNONet(**cfg.MODEL.afno)

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        pretrained_model_path=cfg.INFER.pretrained_model_path,
    )
    # export model
    from paddle.static import InputSpec

    input_spec = [
        {
            key: InputSpec([None, 20, cfg.IMG_H, cfg.IMG_W], "float32", name=key)
            for key in model.input_keys
        },
    ]
    solver.export(input_spec, cfg.INFER.export_path)


def inference(cfg: DictConfig):
    from deploy.python_infer import pinn_predictor

    predictor = pinn_predictor.PINNPredictor(cfg)

    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )

    data = np.load(cfg.INFER_FILE_PATH)
    input_0 = (data[:, 0] - data_mean) / data_std
    all_data = input_0

    for t in range(cfg.INFER.num_timestamps):
        data_t = data[:, t + 1]
        data_t = (data_t - data_mean) / data_std
        all_data = np.concatenate((all_data, data_t), axis=0)

    input_dict = {cfg.MODEL.afno.input_keys[0]: all_data}

    vis_output = predictor.predict(input_dict, cfg.INFER.batch_size)

    vis_dict = {
        store_key: vis_output[infer_key]
        for store_key, infer_key in zip(cfg.MODEL.afno.output_keys, vis_output.keys())
    }

    def output_wind_func(output, data_mean, data_std):
        output = (output * data_std) + data_mean
        wind_data = (output[0] ** 2 + output[1] ** 2) ** 0.5
        return wind_data

    wind_pred = []
    pred_dict = {}
    for i in range(cfg.INFER.num_timestamps):
        hour = (i + 1) * 6
        wind_ = [
            output_wind_func(
                vis_dict[cfg.MODEL.afno.output_keys[0]][i], data_mean, data_std
            )
        ]
        wind_pred.append(wind_)
        pred_dict[f"output_{hour}h"] = np.asarray(wind_)
    output_dict = {cfg.MODEL.afno.output_keys[0]: np.array(wind_pred)}

    wind_pred = []
    target_dict = {}
    for i in range(cfg.INFER.num_timestamps):
        hour = (i + 1) * 6
        wind_ = [(data[0][i][0] ** 2 + data[0][i][1] ** 2) ** 0.5]
        target_dict[f"target_{hour}h"] = np.asarray(wind_)

    vis_dict = {**pred_dict, **target_dict}

    plot_expr_dict = {}
    for hour in range(6, 6 + cfg.INFER.num_timestamps * 6, 6):
        plot_expr_dict.update(
            {
                f"target_{hour}h": lambda d, hour=hour: d[f"target_{hour}h"],
                f"output_{hour}h": lambda d, hour=hour: d[f"output_{hour}h"],
            }
        )

    visualizer_weather = ppsci.visualize.VisualizerWeather(
        vis_dict,
        plot_expr_dict,
        xticks=np.linspace(0, cfg.IMG_W - 1, 13),
        xticklabels=[str(i) for i in range(360, -1, -30)],
        yticks=np.linspace(0, cfg.IMG_H - 1, 7),
        yticklabels=[str(i) for i in range(90, -91, -30)],
        vmin=0,
        vmax=25,
        colorbar_label="m\s",
        batch_size=1,
        num_timestamps=cfg.INFER.num_timestamps,
        prefix="wind",
    )
    visualizer_weather.save(cfg.INFER.export_path, vis_dict)
    save_path = osp.join(cfg.INFER.export_path, "predict.npy")
    os.makedirs(cfg.INFER.export_path, exist_ok=True)
    np.save(save_path, output_dict[cfg.MODEL.afno.output_keys[0]])


@hydra.main(
    version_base=None, config_path="./conf", config_name="fourcastnet_finetune.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == "export":
        export(cfg)
    elif cfg.mode == "infer":
        inference(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
        )


if __name__ == "__main__":
    main()

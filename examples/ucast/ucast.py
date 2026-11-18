# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import paddle
from omegaconf import DictConfig
from ucast_utils import SURFACE_VARIABLES
from ucast_utils import VARIABLE_NAMES
from ucast_utils import UCastERA5Dataset
from ucast_utils import autoregressive_inference
from ucast_utils import build_xarray_dataset
from ucast_utils import ensure_latitude_is_ascending
from ucast_utils import extract_batch_for_initial_condition
from ucast_utils import load_normalizer
from ucast_utils import load_static_conditions
from ucast_utils import load_static_conditions_numpy
from ucast_utils import open_era5_zarr
from ucast_utils import score_forecast

import ppsci
from ppsci.utils import logger
from ppsci.utils import save_load


def train(cfg: DictConfig):
    """Train either stage with the original optimizer and validation semantics."""
    from ucast_training import UCastSolver
    from ucast_training import validate_training_config

    validate_training_config(cfg)
    paddle.seed(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.set_device(cfg.device)
    dataset = ensure_latitude_is_ascending(
        open_era5_zarr(cfg.DATA.data_dir, cfg.DATA.dataset, cfg.DATA.get("cache_dir"))
    )
    normalizer = load_normalizer(cfg.DATA.stats_dir, dataset, VARIABLE_NAMES)
    static_condition = load_static_conditions_numpy(dataset, cfg.DATA.static_fields)
    train_dataset = UCastERA5Dataset(
        dataset,
        normalizer,
        static_condition,
        cfg.TRAIN.train_time_slices,
        window=cfg.DATA.window,
        horizon=cfg.DATA.horizon,
        hourly_resolution=cfg.DATA.hourly_resolution,
        max_samples=cfg.TRAIN.max_train_samples,
    )
    val_dataset = UCastERA5Dataset(
        dataset,
        normalizer,
        static_condition,
        [cfg.TRAIN.val_time_slice],
        window=cfg.DATA.window,
        horizon=cfg.DATA.horizon,
        hourly_resolution=cfg.DATA.hourly_resolution,
        possible_initial_hours=cfg.TRAIN.possible_initial_hours_eval,
        max_samples=cfg.TRAIN.max_val_samples,
        prediction_horizon=cfg.TRAIN.prediction_horizon,
        validation=True,
    )
    model = ppsci.arch.UCast(**cfg.MODEL)
    solver = UCastSolver(
        cfg, model, train_dataset, val_dataset, normalizer, dataset.latitude.values
    )
    solver.train()
    return solver


def infer(cfg: DictConfig):
    """Run released-checkpoint U-Cast inference with PaddlePaddle."""
    paddle.seed(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.set_device(cfg.device)

    model = ppsci.arch.UCast(**cfg.MODEL)
    save_load.load_pretrain(model, cfg.INFER.pretrained_model_path)
    use_amp = bool(cfg.INFER.use_amp) and paddle.get_device().startswith("gpu")
    if cfg.INFER.use_amp and not use_amp:
        logger.warning("U-Cast FP16 inference is only enabled on GPU; using FP32")
    if use_amp:
        model.prepare_for_inference_amp()
    model.enable_mc_dropout()
    logger.info(
        f"Loaded U-Cast with {sum(parameter.numel() for parameter in model.parameters()):,} parameters"
    )

    dataset = ensure_latitude_is_ascending(
        open_era5_zarr(cfg.DATA.data_dir, cfg.DATA.dataset, cfg.DATA.get("cache_dir"))
    )
    logger.info(f"Opened {cfg.DATA.dataset} with {len(dataset.time):,} time steps")
    normalizer = load_normalizer(cfg.DATA.stats_dir, dataset, VARIABLE_NAMES)
    static_condition = load_static_conditions(dataset, cfg.DATA.static_fields)

    initial_conditions = [np.datetime64(value) for value in cfg.INFER.ic_start_dates]
    forecast_datasets = []
    all_scores = []
    for index, initial_condition in enumerate(initial_conditions, start=1):
        logger.info(
            f"Initial condition {index}/{len(initial_conditions)}: {initial_condition}"
        )
        batch = extract_batch_for_initial_condition(
            dataset,
            initial_condition,
            VARIABLE_NAMES,
            normalizer,
            cfg.DATA.window,
            cfg.INFER.prediction_horizon,
            cfg.DATA.hourly_resolution,
        )
        result = autoregressive_inference(
            model,
            batch,
            static_condition,
            normalizer,
            cfg.DATA.window,
            cfg.INFER.prediction_horizon,
            cfg.INFER.ensemble_size,
            use_amp,
        )
        forecast_datasets.append(
            build_xarray_dataset(
                result,
                batch,
                initial_condition,
                cfg.DATA.hourly_resolution,
                VARIABLE_NAMES,
            ).expand_dims("init_time")
        )
        if cfg.INFER.score:
            all_scores.append(score_forecast(result, batch["latitude"], VARIABLE_NAMES))

    if cfg.INFER.score:
        lead_hours = (
            np.arange(1, cfg.INFER.prediction_horizon + 1) * cfg.DATA.hourly_resolution
        )
        for variable in SURFACE_VARIABLES:
            crps = np.mean([score["crps"][variable] for score in all_scores], axis=0)
            rmse = np.mean([score["rmse"][variable] for score in all_scores], axis=0)
            logger.info(f"Scores for {variable}")
            for lead_hour, crps_value, rmse_value in zip(lead_hours, crps, rmse):
                logger.info(
                    f"  {lead_hour:3d}h: CRPS={crps_value:.4f}, RMSE={rmse_value:.4f}"
                )

    if cfg.INFER.output_path:
        import xarray as xr

        output_path = Path(cfg.INFER.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xr.concat(forecast_datasets, dim="init_time").to_netcdf(output_path)
        logger.info(f"Saved forecasts to {output_path}")
    return forecast_datasets


@hydra.main(version_base=None, config_path="./conf", config_name="ucast.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "infer":
        infer(cfg)
    else:
        raise ValueError(f"cfg.mode should be 'train' or 'infer', but got {cfg.mode!r}")


if __name__ == "__main__":
    main()

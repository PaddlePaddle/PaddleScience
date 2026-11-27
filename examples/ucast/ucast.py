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
from ucast_utils import (
    SURFACE_VARIABLES,
    VARIABLE_NAMES,
    autoregressive_inference,
    build_xarray_dataset,
    ensure_latitude_is_ascending,
    extract_batch_for_initial_condition,
    load_normalizer,
    load_static_conditions,
    open_era5_zarr,
    score_forecast,
)

import ppsci
from ppsci.utils import logger


def infer(cfg: DictConfig):
    """MC-Dropout and deep-ensemble inference with sequential weight loading."""
    import gc
    import json

    from ucast_training import load_weights

    paddle.seed(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.set_device(cfg.device)
    paths = cfg.INFER.get("pretrained_model_paths") or [cfg.INFER.pretrained_model_path]
    if isinstance(paths, str):
        paths = [paths]
    use_amp = bool(cfg.INFER.use_amp) and paddle.get_device().startswith("gpu")
    dataset = ensure_latitude_is_ascending(
        open_era5_zarr(cfg.DATA.data_dir, cfg.DATA.dataset, cfg.DATA.get("cache_dir"))
    )
    normalizer = load_normalizer(cfg.DATA.stats_dir, dataset, VARIABLE_NAMES)
    static_condition = load_static_conditions(dataset, cfg.DATA.static_fields)
    initial_conditions = [np.datetime64(value) for value in cfg.INFER.ic_start_dates]
    if not initial_conditions:
        raise ValueError("INFER.ic_start_dates must contain at least one timestamp")
    base_members, remainder = divmod(int(cfg.INFER.ensemble_size), len(paths))
    members_per_model = [
        base_members + (index < remainder) for index in range(len(paths))
    ]
    if min(members_per_model) < 1:
        raise ValueError(
            "INFER.ensemble_size must be at least the number of checkpoints"
        )
    per_initial_condition = [[] for _ in initial_conditions]
    for model_index, path in enumerate(paths):
        model = ppsci.arch.UCast(**cfg.MODEL)
        load_weights(model, path, use_ema=True)
        if use_amp:
            model.prepare_for_inference_amp()
        model.enable_mc_dropout()
        logger.info(f"Loaded ensemble model {model_index + 1}/{len(paths)}: {path}")
        for index, initial_condition in enumerate(initial_conditions):
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
                members_per_model[model_index],
                use_amp,
                cfg.INFER.get("ensemble_chunk_size"),
            )
            per_initial_condition[index].append(result)
        del model
        gc.collect()
        if paddle.get_device().startswith("gpu"):
            paddle.device.cuda.empty_cache()
    forecast_datasets, all_scores = [], []
    for index, initial_condition in enumerate(initial_conditions):
        parts = per_initial_condition[index]
        result = {
            "predictions": paddle.concat(
                [part["predictions"] for part in parts], axis=0
            ),
            "targets": parts[0]["targets"],
        }
        batch_coords = {
            "longitude": dataset.longitude.values,
            "latitude": dataset.latitude.values,
        }
        forecast = build_xarray_dataset(
            result,
            batch_coords,
            initial_condition,
            cfg.DATA.hourly_resolution,
            VARIABLE_NAMES,
        )
        forecast = forecast.assign_coords(
            source_model=(
                "ensemble_member",
                np.repeat(np.arange(len(paths)), members_per_model),
            ),
            mc_member=(
                "ensemble_member",
                np.concatenate([np.arange(count) for count in members_per_model]),
            ),
        )
        forecast.attrs["checkpoint_paths"] = json.dumps([str(path) for path in paths])
        forecast_datasets.append(forecast.expand_dims("init_time"))
        if cfg.INFER.score:
            all_scores.append(
                score_forecast(result, dataset.latitude.values, VARIABLE_NAMES)
            )
    if cfg.INFER.score:
        scores = {
            metric: {
                name: np.mean(
                    [score[metric][name] for score in all_scores], axis=0
                ).tolist()
                for name in VARIABLE_NAMES
            }
            for metric in ("rmse", "crps")
        }
        for name in SURFACE_VARIABLES:
            logger.info(
                f"{name}: RMSE={scores['rmse'][name]}, fair CRPS={scores['crps'][name]}"
            )
        if cfg.INFER.get("score_output_path"):
            score_path = Path(cfg.INFER.score_output_path)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps(scores, indent=2))
    if cfg.INFER.output_path:
        import xarray as xr

        output_path = Path(cfg.INFER.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xr.concat(forecast_datasets, dim="init_time").to_netcdf(
            output_path, encoding={"lead_time": {"dtype": "int64", "units": "hours"}}
        )
        logger.info(f"Saved forecasts to {output_path}")
    return forecast_datasets


@hydra.main(version_base=None, config_path="./conf", config_name="ucast.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "infer":
        infer(cfg)
    elif cfg.mode == "eval":
        from ucast_training import evaluate_checkpoint

        evaluate_checkpoint(cfg)
    elif cfg.mode == "train":
        from ucast_training import train

        train(cfg)
    else:
        raise ValueError(f"Unknown U-Cast mode: {cfg.mode!r}")


if __name__ == "__main__":
    main()

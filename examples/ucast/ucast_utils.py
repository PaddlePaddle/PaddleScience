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

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import paddle

SURFACE_VARIABLES = (
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
)
PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
UPPER_AIR_VARIABLES = (
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
)
VARIABLE_NAMES = SURFACE_VARIABLES + tuple(
    f"{variable}_{level}"
    for variable in UPPER_AIR_VARIABLES
    for level in PRESSURE_LEVELS
)

SECONDS_PER_DAY = 86400
AVERAGE_DAYS_PER_YEAR = 365.24219
GRAPHCAST_PRESSURE_LEVELS = (
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
)


def ensure_latitude_is_ascending(dataset):
    """Return an xarray object whose latitude coordinate is ascending."""
    if not (np.diff(dataset.latitude.values) > 0).all():
        return dataset.reindex(latitude=list(reversed(dataset.latitude.values)))
    return dataset


def open_era5_zarr(
    data_dir: str, dataset_name: str, cache_dir: str | Path | None = None
):
    """Open a local or public-GCS WeatherBench-2 Zarr dataset lazily."""
    import xarray as xr

    path = os.path.join(data_dir, dataset_name)
    if os.path.isdir(path):
        return xr.open_zarr(path)
    if data_dir.startswith("gs://"):
        if cache_dir:
            from era5_cache import ERA5Cache

            return xr.open_zarr(ERA5Cache(path, cache_dir), consolidated=True)
        import gcsfs

        filesystem = gcsfs.GCSFileSystem(token="anon")
        return xr.open_zarr(filesystem.get_mapper(path), zarr_format=2)
    raise FileNotFoundError(f"Cannot find ERA5 dataset at {path}")


def _extract_stat(dataset, name: str) -> np.ndarray:
    if name in dataset:
        return np.asarray(dataset[name].values, dtype=np.float32)
    variable, level = name.rsplit("_", 1)
    return np.asarray(dataset[variable].sel(level=int(level)).values, dtype=np.float32)


class UCastNormalizer:
    """Packed-tensor normalization used by the released U-Cast checkpoint."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        residual_std: np.ndarray,
        sst_nan_mask: np.ndarray,
        sst_fill_value: float,
        variable_names: Sequence[str] = VARIABLE_NAMES,
    ):
        self.variable_names = tuple(variable_names)
        self.sst_index = self.variable_names.index("sea_surface_temperature")
        self.mean_numpy = mean.reshape([1, -1, 1, 1]).astype(np.float32)
        self.std_numpy = std.reshape([1, -1, 1, 1]).astype(np.float32)
        self.residual_scale_numpy = (
            (residual_std / std).reshape([1, -1, 1, 1]).astype(np.float32)
        )
        self.target_residual_scale_numpy = 1.0 / self.residual_scale_numpy
        self.mean = paddle.to_tensor(self.mean_numpy)
        self.std = paddle.to_tensor(self.std_numpy)
        self.residual_scale = paddle.to_tensor(self.residual_scale_numpy)
        self.sst_nan_mask_numpy = np.asarray(sst_nan_mask, dtype=bool)
        self.sst_nan_mask = paddle.to_tensor(self.sst_nan_mask_numpy, dtype="bool")
        self.sst_fill_value = float(sst_fill_value)

    def _fill_sst(self, tensor: paddle.Tensor) -> paddle.Tensor:
        channels = list(paddle.unstack(tensor, axis=-3))
        channels[self.sst_index] = paddle.where(
            self.sst_nan_mask,
            paddle.full_like(channels[self.sst_index], self.sst_fill_value),
            channels[self.sst_index],
        )
        return paddle.stack(channels, axis=-3)

    def normalize(self, raw: paddle.Tensor) -> paddle.Tensor:
        return (self._fill_sst(raw) - self.mean) / self.std

    def normalize_numpy(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32).copy()
        raw[..., self.sst_index, :, :] = np.where(
            self.sst_nan_mask_numpy,
            self.sst_fill_value,
            raw[..., self.sst_index, :, :],
        )
        return (raw - self.mean_numpy) / self.std_numpy

    def normalized_to_residual_numpy(self, residual: np.ndarray) -> np.ndarray:
        return residual * self.target_residual_scale_numpy

    def denormalize(self, normalized: paddle.Tensor) -> paddle.Tensor:
        return self._fill_sst(normalized * self.std + self.mean)

    def residual_to_normalized(self, residual: paddle.Tensor) -> paddle.Tensor:
        return residual.astype(self.residual_scale.dtype) * self.residual_scale


def load_normalizer(
    stats_dir: str | Path,
    dataset,
    variable_names: Sequence[str] = VARIABLE_NAMES,
) -> UCastNormalizer:
    """Load the three released NetCDF statistics files and the SST mask."""
    import xarray as xr

    stats_dir = Path(stats_dir)
    with xr.open_dataset(stats_dir / "era5_mean.nc") as mean_ds:
        mean = np.stack([_extract_stat(mean_ds, name) for name in variable_names])
    with xr.open_dataset(stats_dir / "era5_std.nc") as std_ds:
        std = np.stack([_extract_stat(std_ds, name) for name in variable_names])
    with xr.open_dataset(stats_dir / "era5_residual_std.nc") as residual_std_ds:
        residual_std = np.stack(
            [_extract_stat(residual_std_ds, name) for name in variable_names]
        )
    with xr.open_dataset(stats_dir / "era5_min.nc") as min_ds:
        sst_fill_value = float(min_ds["sea_surface_temperature"].values.item())

    sst_sample = ensure_latitude_is_ascending(
        dataset["sea_surface_temperature"].isel(time=0).compute()
    ).transpose("longitude", "latitude")
    sst_nan_mask = np.isnan(sst_sample.values)
    if sst_nan_mask.mean() >= 0.5:
        raise ValueError("More than half of the SST grid is NaN; the data look corrupt")

    return UCastNormalizer(
        mean,
        std,
        residual_std,
        sst_nan_mask,
        sst_fill_value,
        variable_names,
    )


def load_static_conditions_numpy(dataset, static_fields: Sequence[str]) -> np.ndarray:
    """Load and independently z-score U-Cast's two static fields."""
    arrays = []
    for field in static_fields:
        data_array = dataset[field]
        if "time" in data_array.dims:
            data_array = data_array.isel(time=0)
        values = data_array.transpose("longitude", "latitude").compute().values
        arrays.append(np.asarray(values, dtype=np.float32))
    stacked = np.stack(arrays)
    mean = stacked.mean(axis=(-2, -1), keepdims=True)
    std = stacked.std(axis=(-2, -1), keepdims=True)
    return ((stacked - mean) / std).astype(np.float32)


def load_static_conditions(dataset, static_fields: Sequence[str]) -> paddle.Tensor:
    return paddle.to_tensor(load_static_conditions_numpy(dataset, static_fields))


def compute_forcings_numpy(
    time_values: np.ndarray,
    longitude: np.ndarray,
    spatial_shape: tuple[int, int],
) -> np.ndarray:
    """Compute year/day progress sine and cosine fields."""
    seconds = time_values.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    year_progress = np.mod(
        seconds / SECONDS_PER_DAY / AVERAGE_DAYS_PER_YEAR, 1.0
    ).astype(np.float32)
    day_fraction = np.mod(seconds, SECONDS_PER_DAY) / SECONDS_PER_DAY
    longitude_offset = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(
        day_fraction[:, None] + longitude_offset[None, :], 1.0
    ).astype(np.float32)

    num_longitude, num_latitude = spatial_shape
    forcing = np.stack(
        (
            np.broadcast_to(
                np.sin(2 * np.pi * year_progress)[:, None, None],
                (len(time_values), num_longitude, num_latitude),
            ),
            np.broadcast_to(
                np.cos(2 * np.pi * year_progress)[:, None, None],
                (len(time_values), num_longitude, num_latitude),
            ),
            np.broadcast_to(
                np.sin(2 * np.pi * day_progress)[:, :, None],
                (len(time_values), num_longitude, num_latitude),
            ),
            np.broadcast_to(
                np.cos(2 * np.pi * day_progress)[:, :, None],
                (len(time_values), num_longitude, num_latitude),
            ),
        ),
        axis=1,
    ).astype(np.float32)
    return forcing


def compute_forcings(
    time_values: np.ndarray,
    longitude: np.ndarray,
    spatial_shape: tuple[int, int],
) -> paddle.Tensor:
    return paddle.to_tensor(
        compute_forcings_numpy(time_values, longitude, spatial_shape)
    )


def _variables_needed_from_zarr(dataset, variable_names: Sequence[str]) -> list[str]:
    variables = []
    for name in variable_names:
        source_name = name if name in dataset else name.rsplit("_", 1)[0]
        if source_name not in variables:
            variables.append(source_name)
    return variables


def _infer_time_subsample(dataset, hourly_resolution: int) -> int:
    if len(dataset.time) < 2:
        raise ValueError("At least two ERA5 time steps are required")
    delta_hours = int(
        (dataset.time.values[1] - dataset.time.values[0]) / np.timedelta64(1, "h")
    )
    if delta_hours <= 0 or hourly_resolution % delta_hours:
        raise ValueError(
            f"Cannot sample {hourly_resolution}h data from {delta_hours}h source data"
        )
    return hourly_resolution // delta_hours


def _latitude_cell_weights(latitude: np.ndarray) -> np.ndarray:
    points = np.deg2rad(np.asarray(latitude))
    bounds = np.concatenate(
        [
            np.asarray([-np.pi / 2], dtype=points.dtype),
            (points[:-1] + points[1:]) / 2,
            np.asarray([np.pi / 2], dtype=points.dtype),
        ]
    )
    if not np.all(np.diff(bounds) > 0):
        raise ValueError("Latitude coordinates must be strictly increasing")
    weights = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    return (weights / weights.mean()).astype(np.float32)


def build_loss_weights(
    latitude: np.ndarray,
    variable_names: Sequence[str] = VARIABLE_NAMES,
) -> np.ndarray:
    """Build U-Cast's latitude-area and GraphCast variable loss weights."""
    variable_weights = np.ones(len(variable_names), dtype=np.float32)
    pressure_divisor = float(np.mean(GRAPHCAST_PRESSURE_LEVELS))
    surface_weights = {
        "2m_temperature": 1.0,
        "10m_u_component_of_wind": 0.1,
        "10m_v_component_of_wind": 0.1,
        "mean_sea_level_pressure": 0.1,
        "sea_surface_temperature": 0.1,
    }
    for index, name in enumerate(variable_names):
        if name in surface_weights:
            variable_weights[index] = surface_weights[name]
        else:
            variable_weights[index] = int(name.rsplit("_", 1)[1]) / pressure_divisor
    return (
        variable_weights[:, None, None]
        * _latitude_cell_weights(latitude)[None, None, :]
    ).astype(np.float32)


class UCastERA5Dataset(paddle.io.Dataset):
    """Lazy single-step ERA5 dataset matching the original U-Cast training path."""

    input_keys = ("input", "dynamical_condition", "static_condition")
    label_keys = ("output",)

    def __init__(
        self,
        dataset,
        normalizer: UCastNormalizer,
        static_condition: np.ndarray,
        time_slices: Sequence[Sequence[str]],
        variable_names: Sequence[str] = VARIABLE_NAMES,
        window: int = 2,
        horizon: int = 1,
        hourly_resolution: int = 12,
        possible_initial_hours: Sequence[int] | None = None,
        max_samples: int | None = None,
        prediction_horizon: int = 1,
        validation: bool = False,
    ):
        super().__init__()
        if horizon != 1:
            raise ValueError("U-Cast training currently supports horizon=1 only")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        self.validation = validation
        self.prediction_horizon = prediction_horizon
        self.normalizer = normalizer
        self.static_condition = np.asarray(static_condition, dtype=np.float32)
        self.variable_names = tuple(variable_names)
        self.window = window
        self.horizon = horizon
        self.source_variables = _variables_needed_from_zarr(
            dataset, self.variable_names
        )
        self.requested_levels = sorted(
            {
                int(name.rsplit("_", 1)[1])
                for name in self.variable_names
                if name not in dataset
            }
        )
        stride = _infer_time_subsample(dataset, hourly_resolution)
        self.segments = []
        sample_indices = []
        allowed_hours = (
            None
            if possible_initial_hours is None
            else {int(hour) for hour in possible_initial_hours}
        )
        for start, stop in time_slices:
            segment = dataset[self.source_variables].sel(time=slice(start, stop))
            segment = segment.isel(time=slice(None, None, stride))
            if "level" in segment.dims:
                segment = segment.sel(level=self.requested_levels)
            segment_index = len(self.segments)
            self.segments.append(segment)
            num_samples = len(segment.time) - window - prediction_horizon + 1
            for start_index in range(max(0, num_samples)):
                initial_time = segment.time.values[start_index + window - 1]
                initial_hour = int(
                    initial_time.astype("datetime64[h]").astype(np.int64) % 24
                )
                if allowed_hours is None or initial_hour in allowed_hours:
                    sample_indices.append((segment_index, start_index))
        if max_samples is not None and len(sample_indices) > max_samples:
            selected = np.linspace(
                0, len(sample_indices) - 1, max_samples, dtype=np.int64
            )
            sample_indices = [sample_indices[index] for index in selected]
        self.sample_indices = sample_indices
        if not self.sample_indices:
            raise ValueError(
                "No valid U-Cast samples were found for the requested split"
            )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        segment_index, start_index = self.sample_indices[index]
        batch = self.segments[segment_index].isel(
            time=slice(start_index, start_index + self.window + self.prediction_horizon)
        )
        batch = batch.load()
        arrays = []
        for name in self.variable_names:
            if name in batch:
                data_array = batch[name]
            else:
                variable, level = name.rsplit("_", 1)
                data_array = batch[variable].sel(level=int(level))
            arrays.append(
                np.asarray(
                    data_array.transpose("time", "longitude", "latitude").values,
                    dtype=np.float32,
                )
            )
        dynamics = self.normalizer.normalize_numpy(np.stack(arrays, axis=1))
        target = self.normalizer.normalized_to_residual_numpy(
            dynamics[self.window] - dynamics[self.window - 1]
        )[0]
        longitude = batch.longitude.values
        forcing = compute_forcings_numpy(
            batch.time.values[self.window : self.window + 1],
            longitude,
            (len(longitude), len(batch.latitude)),
        )[0]
        inputs = {
            "input": dynamics[: self.window].reshape(
                [-1, dynamics.shape[-2], dynamics.shape[-1]]
            ),
            "dynamical_condition": forcing,
            "static_condition": self.static_condition,
        }
        if self.validation:
            inputs["forecast_forcing"] = compute_forcings_numpy(
                batch.time.values[self.window :],
                longitude,
                (len(longitude), len(batch.latitude)),
            )
            return inputs, {"output": dynamics[self.window :]}, {}
        return inputs, {"output": target}, {}


def weighted_mae_loss(loss_weights: paddle.Tensor):
    def loss_fn(output_dict, label_dict, weight_dict=None):
        prediction = output_dict["output"]
        target = label_dict["output"]
        return {"wmae": paddle.mean(paddle.abs(prediction - target) * loss_weights)}

    return loss_fn


def weighted_crps_loss(loss_weights: paddle.Tensor):
    def loss_fn(output_dict, label_dict, weight_dict=None):
        prediction = output_dict["output"]
        target = label_dict["output"]
        if prediction.ndim != target.ndim + 1 or prediction.shape[0] < 2:
            raise ValueError("Fair CRPS requires at least two ensemble members")
        num_members = prediction.shape[0]
        skill = paddle.mean(paddle.abs(prediction - target.unsqueeze(0)), axis=0)
        pairwise = paddle.abs(prediction.unsqueeze(0) - prediction.unsqueeze(1))
        spread = paddle.sum(pairwise, axis=(0, 1)) / (num_members * (num_members - 1))
        return {"wcrps": paddle.mean((skill - 0.5 * spread) * loss_weights)}

    return loss_fn


def residual_rmse_metric(output_dict, label_dict):
    prediction = output_dict["output"]
    if prediction.ndim == label_dict["output"].ndim + 1:
        prediction = paddle.mean(prediction, axis=0)
    return {
        "residual_rmse": paddle.sqrt(
            paddle.mean((prediction - label_dict["output"]) ** 2)
        )
    }


def extract_batch_for_initial_condition(
    dataset,
    initial_condition: np.datetime64,
    variable_names: Sequence[str],
    normalizer: UCastNormalizer,
    window: int,
    prediction_horizon: int,
    hourly_resolution: int,
) -> dict[str, object]:
    """Load only the variables and times needed for one rollout."""
    start = initial_condition - np.timedelta64((window - 1) * hourly_resolution, "h")
    num_steps = window + prediction_horizon
    target_times = np.asarray(
        [
            start + np.timedelta64(step * hourly_resolution, "h")
            for step in range(num_steps)
        ],
        dtype=dataset.time.values.dtype,
    )
    source_variables = _variables_needed_from_zarr(dataset, variable_names)
    batch = dataset[source_variables].sel(time=target_times)
    if "level" in batch.dims:
        requested_levels = sorted(
            {
                int(name.rsplit("_", 1)[1])
                for name in variable_names
                if name not in dataset
            }
        )
        batch = batch.sel(level=requested_levels)
    batch = batch.load()
    if len(batch.time) != num_steps:
        raise ValueError(
            f"Requested {num_steps} time steps from {start}, got {len(batch.time)}"
        )

    arrays = []
    for name in variable_names:
        if name in batch:
            data_array = batch[name]
        else:
            variable, level = name.rsplit("_", 1)
            data_array = batch[variable].sel(level=int(level))
        arrays.append(
            np.asarray(
                data_array.transpose("time", "longitude", "latitude").values,
                dtype=np.float32,
            )
        )
    dynamics_raw = paddle.to_tensor(np.stack(arrays, axis=1))
    dynamics_normalized = normalizer.normalize(dynamics_raw)
    longitude = batch.longitude.values
    latitude = batch.latitude.values
    forcing = compute_forcings(
        batch.time.values,
        longitude,
        (len(longitude), len(latitude)),
    )
    return {
        "dynamics_raw": dynamics_raw,
        "dynamics_normalized": dynamics_normalized,
        "forcing": forcing,
        "time": batch.time.values,
        "longitude": longitude,
        "latitude": latitude,
    }


@paddle.no_grad()
def autoregressive_inference(
    model,
    batch: dict[str, object],
    static_condition: paddle.Tensor,
    normalizer: UCastNormalizer,
    window: int,
    prediction_horizon: int,
    ensemble_size: int,
    use_amp: bool = True,
) -> dict[str, paddle.Tensor]:
    """Run an MC-Dropout ensemble autoregressive forecast."""
    frames = [
        batch["dynamics_normalized"][index]
        .unsqueeze(0)
        .expand([ensemble_size, -1, -1, -1])
        for index in range(window)
    ]
    static_batch = static_condition.unsqueeze(0).expand([ensemble_size, -1, -1, -1])
    predictions = []
    targets = []
    for lead_time in range(prediction_horizon):
        forcing_index = window + lead_time
        forcing = (
            batch["forcing"][forcing_index]
            .unsqueeze(0)
            .expand([ensemble_size, -1, -1, -1])
        )
        inputs = paddle.concat(frames, axis=1)
        with paddle.amp.auto_cast(
            enable=use_amp,
            level="O2",
            dtype="float16",
        ):
            residual = model(
                {
                    "input": inputs,
                    "dynamical_condition": forcing,
                    "static_condition": static_batch,
                }
            )["output"]
        if bool(paddle.any(paddle.isnan(residual))):
            raise FloatingPointError(f"NaN in model output at lead {lead_time + 1}")
        prediction_normalized = normalizer.residual_to_normalized(residual) + frames[-1]
        predictions.append(normalizer.denormalize(prediction_normalized).cpu())
        targets.append(batch["dynamics_raw"][forcing_index].cpu())
        frames = frames[1:] + [prediction_normalized]

    return {
        "predictions": paddle.stack(predictions, axis=1),
        "targets": paddle.stack(targets),
    }


def build_xarray_dataset(
    result: dict[str, paddle.Tensor],
    batch: dict[str, object],
    initial_condition: np.datetime64,
    hourly_resolution: int,
    variable_names: Sequence[str] = VARIABLE_NAMES,
):
    """Convert packed forecast/target tensors into a labelled xarray Dataset."""
    import xarray as xr

    predictions = result["predictions"].numpy()
    targets = result["targets"].numpy()
    lead_time = np.arange(1, predictions.shape[1] + 1) * np.timedelta64(
        hourly_resolution, "h"
    )
    data_vars = {}
    for index, name in enumerate(variable_names):
        data_vars[f"{name}_predicted"] = xr.DataArray(
            predictions[:, :, index],
            dims=("ensemble_member", "lead_time", "longitude", "latitude"),
            coords={
                "ensemble_member": np.arange(predictions.shape[0]),
                "lead_time": lead_time,
                "longitude": batch["longitude"],
                "latitude": batch["latitude"],
            },
        )
        data_vars[f"{name}_target"] = xr.DataArray(
            targets[:, index],
            dims=("lead_time", "longitude", "latitude"),
            coords={
                "lead_time": lead_time,
                "longitude": batch["longitude"],
                "latitude": batch["latitude"],
            },
        )
    return xr.Dataset(data_vars).assign_coords(init_time=initial_condition)


def score_forecast(
    result: dict[str, paddle.Tensor],
    latitude: np.ndarray,
    variable_names: Sequence[str] = VARIABLE_NAMES,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute latitude-weighted fair CRPS and ensemble-mean RMSE."""
    predictions = result["predictions"]
    targets = result["targets"]
    weights = paddle.to_tensor(
        np.cos(np.deg2rad(latitude)).astype(np.float32), place=predictions.place
    )
    weights /= weights.mean()
    crps = {}
    rmse = {}
    for index, name in enumerate(variable_names):
        prediction = predictions[:, :, index]
        target = targets[:, index]
        term1 = paddle.mean(paddle.abs(prediction - target.unsqueeze(0)), axis=0)
        num_members = prediction.shape[0]
        if num_members > 1:
            pairwise = paddle.abs(prediction.unsqueeze(0) - prediction.unsqueeze(1))
            term2 = paddle.sum(pairwise, axis=(0, 1)) / (
                num_members * (num_members - 1)
            )
        else:
            term2 = paddle.zeros_like(term1)
        crps[name] = paddle.nanmean(
            (term1 - 0.5 * term2) * weights[None, None, :], axis=(-2, -1)
        ).numpy()
        ensemble_mean = paddle.mean(prediction, axis=0)
        rmse[name] = paddle.sqrt(
            paddle.nanmean(
                (ensemble_mean - target) ** 2 * weights[None, None, :],
                axis=(-2, -1),
            )
        ).numpy()
    return {"crps": crps, "rmse": rmse}

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Lazy ERA5 windows and the original U-Cast training weights."""
from __future__ import annotations

import numpy as np
import paddle
from ucast_utils import (
    SURFACE_VARIABLES,
    VARIABLE_NAMES,
    _variables_needed_from_zarr,
    compute_forcings_numpy,
)


def latitude_area_weights(latitude):
    latitude = np.asarray(latitude, dtype=np.float64)
    if latitude.ndim != 1 or len(latitude) < 2 or not np.all(np.diff(latitude) > 0):
        raise ValueError(
            "ERA5 latitude must be a strictly ascending one-dimensional coordinate"
        )
    radians = np.deg2rad(latitude)
    boundaries = np.concatenate(
        ([-np.pi / 2], (radians[1:] + radians[:-1]) / 2, [np.pi / 2])
    )
    area = np.diff(np.sin(boundaries))
    return (area / area.mean()).astype(np.float32)


def training_weights(latitude, variable_names=VARIABLE_NAMES):
    all_levels = (
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
    pressure_mean = np.mean(all_levels)
    variable_weights = [
        (1.0 if name == "2m_temperature" else 0.1)
        if name in SURFACE_VARIABLES
        else int(name.rsplit("_", 1)[1]) / pressure_mean
        for name in variable_names
    ]
    return (
        np.asarray(variable_weights, np.float32)[:, None, None]
        * latitude_area_weights(latitude)[None, None, :]
    )


def select_initial_conditions(
    dataset,
    slices,
    window,
    horizon,
    hourly_resolution,
    possible_initial_hours=None,
    max_samples=None,
):
    """Keep complete windows inside each independent time stream.

    Offsets in the original two train slices (00h and 06h) remain separate;
    windows can never cross a split boundary or a missing timestamp.
    """
    if window < 1 or horizon < 1 or hourly_resolution < 1:
        raise ValueError("window, horizon and hourly_resolution must be positive")
    times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    if len(times) < 2 or not np.all(np.diff(times).astype(np.int64) > 0):
        raise ValueError(
            "ERA5 time must be unique, sorted and have at least two entries"
        )
    step = np.timedelta64(int(hourly_resolution), "h").astype("timedelta64[ns]")
    base_step = np.min(np.diff(times))
    if step % base_step != np.timedelta64(0, "ns"):
        raise ValueError(
            "Requested hourly resolution must be a multiple of the dataset cadence"
        )
    stride = int(step / base_step)
    if stride < 1:
        raise ValueError("Requested hourly resolution is finer than the ERA5 data")
    if not slices:
        slices = [(str(times[0]), str(times[-1]))]
    selected = []
    for begin, end in slices:
        stream = times[(times >= np.datetime64(begin)) & (times <= np.datetime64(end))][
            ::stride
        ]
        n = len(stream) - window - horizon + 1
        for start in range(max(0, n)):
            chunk = stream[start : start + window + horizon]
            if not np.all(np.diff(chunk) == step):
                continue
            initial = chunk[window - 1]
            hour = int(initial.astype("datetime64[h]").astype(np.int64) % 24)
            if possible_initial_hours is None or hour in possible_initial_hours:
                selected.append(initial)
    selected = np.asarray(selected, dtype="datetime64[ns]")
    if max_samples is not None:
        if int(max_samples) < 1:
            raise ValueError("max_samples must be positive")
        if len(selected) > max_samples:
            selected = selected[
                np.linspace(0, len(selected) - 1, int(max_samples), dtype=int)
            ]
    if not len(selected):
        raise ValueError("No complete ERA5 windows found in the configured date ranges")
    return selected


class ERA5WindowDataset(paddle.io.Dataset):
    """Load only one window at a time; workers return CPU NumPy arrays."""

    def __init__(
        self,
        dataset,
        normalizer,
        static_condition,
        initial_conditions,
        window=2,
        horizon=1,
        hourly_resolution=12,
        variable_names=VARIABLE_NAMES,
    ):
        super().__init__()
        self.dataset = dataset[_variables_needed_from_zarr(dataset, variable_names)]
        levels = sorted(
            {int(n.rsplit("_", 1)[1]) for n in variable_names if n not in dataset}
        )
        if levels:
            self.dataset = self.dataset.sel(level=levels)
        self.initial_conditions = np.asarray(initial_conditions).astype(
            "datetime64[ns]"
        )
        self.window, self.horizon, self.hourly_resolution = (
            window,
            horizon,
            hourly_resolution,
        )
        self.variable_names = tuple(variable_names)
        self.mean = normalizer.mean.cpu().numpy()[0]
        self.std = normalizer.std.cpu().numpy()[0]
        self.residual_scale = normalizer.residual_scale.cpu().numpy()[0]
        self.sst_index = normalizer.sst_index
        self.sst_nan_mask = normalizer.sst_nan_mask.cpu().numpy()
        self.sst_fill_value = normalizer.sst_fill_value
        self.static_condition = static_condition.cpu().numpy().astype(np.float32)
        self.longitude = dataset.longitude.values
        self.latitude = dataset.latitude.values

    def __len__(self):
        return len(self.initial_conditions)

    def __getitem__(self, index):
        initial = self.initial_conditions[index]
        times = initial + np.arange(1 - self.window, self.horizon + 1) * np.timedelta64(
            self.hourly_resolution, "h"
        )
        batch = self.dataset.sel(time=times).load()
        arrays = []
        for name in self.variable_names:
            if name in batch:
                field = batch[name]
            else:
                stem, level = name.rsplit("_", 1)
                field = batch[stem].sel(level=int(level))
            arrays.append(
                np.asarray(
                    field.transpose("time", "longitude", "latitude").values, np.float32
                )
            )
        raw = np.stack(arrays, axis=1)
        raw[:, self.sst_index] = np.where(
            self.sst_nan_mask, self.sst_fill_value, raw[:, self.sst_index]
        )
        if not np.isfinite(raw).all():
            raise ValueError(f"Non-finite ERA5 data outside the SST mask at {initial}")
        normalized = (raw - self.mean) / self.std
        forcing = compute_forcings_numpy(times, self.longitude, raw.shape[-2:])
        return {
            "input": normalized[: self.window].reshape((-1, *raw.shape[-2:])),
            "dynamical_condition": forcing[self.window],
            "static_condition": self.static_condition,
            "target": (normalized[self.window] - normalized[self.window - 1])
            / self.residual_scale,
            "dynamics_normalized": normalized,
            "dynamics_raw": raw,
            "forcing": forcing,
        }

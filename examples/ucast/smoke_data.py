# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Make small, explicitly synthetic ERA5-shaped data for integration checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from ucast_utils import PRESSURE_LEVELS, SURFACE_VARIABLES, UPPER_AIR_VARIABLES


def make_smoke_data(output, nlon=8, nlat=5, ntimes=40):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    coordinates = {
        "time": np.datetime64("2000-01-01T00")
        + np.arange(ntimes) * np.timedelta64(6, "h"),
        "longitude": np.linspace(0, 360, nlon, endpoint=False),
        "latitude": np.linspace(90, -90, nlat),
        "level": list(PRESSURE_LEVELS),
    }
    variables, statistics = {}, {
        name: {} for name in ("mean", "std", "residual_std", "min")
    }
    for index, name in enumerate((*SURFACE_VARIABLES, *UPPER_AIR_VARIABLES)):
        levels = () if name in SURFACE_VARIABLES else (len(PRESSURE_LEVELS),)
        shape = (ntimes, *levels, nlon, nlat)
        mean = np.full(levels or (), index + 1.0, np.float32)
        std = np.full(levels or (), 2.0 + index * 0.1, np.float32)
        residual_std = std * 0.2
        values = (rng.normal(size=shape) * 0.05).astype(np.float32)
        values = (
            mean.reshape((1, *levels, 1, 1)) + std.reshape((1, *levels, 1, 1)) * values
        )
        if name == "sea_surface_temperature":
            values[:, 0, 0] = np.nan
        dims = ("time", *(("level",) if levels else ()), "longitude", "latitude")
        variables[name] = (dims, values)
        statdims = ("level",) if levels else ()
        for statname, array in (
            ("mean", mean),
            ("std", std),
            ("residual_std", residual_std),
            ("min", mean - 3 * std),
        ):
            statistics[statname][name] = (statdims, array)
    for name in ("land_sea_mask", "geopotential_at_surface"):
        variables[name] = (
            ("longitude", "latitude"),
            rng.normal(size=(nlon, nlat)).astype(np.float32),
        )
    dataset = xr.Dataset(variables, coords=coordinates)
    path = output / "era5.zarr"
    dataset.chunk({"time": 1}).to_zarr(path, mode="w", zarr_format=2)
    stats = output / "stats"
    stats.mkdir(exist_ok=True)
    for name, fields in statistics.items():
        xr.Dataset(fields, coords={"level": list(PRESSURE_LEVELS)}).to_netcdf(
            stats / f"era5_{name}.nc"
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nlon", type=int, default=8)
    parser.add_argument("--nlat", type=int, default=5)
    parser.add_argument("--ntimes", type=int, default=40)
    args = parser.parse_args()
    print(make_smoke_data(args.output, args.nlon, args.nlat, args.ntimes))

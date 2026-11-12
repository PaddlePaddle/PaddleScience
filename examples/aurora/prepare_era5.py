"""Validate a WeatherBench2 ERA5 subset and produce Aurora Batch NetCDFs."""

import argparse
import json
from pathlib import Path

import numpy as np
import paddle
import xarray as xr
from netcdf_io import write_batch

from ppsci.arch.aurora import Batch
from ppsci.arch.aurora import Metadata

SURFACE = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
STATIC = {"lsm": "land_sea_mask", "slt": "soil_type", "z": "geopotential_at_surface"}
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--netcdf-compression", type=int, choices=range(10), default=0)
    args = parser.parse_args()
    paddle.set_device("cpu")
    root = args.directory
    with xr.open_dataset(root / "surface.nc") as surf, xr.open_dataset(
        root / "atmospheric.nc"
    ) as atmos, xr.open_dataset(root / "static.nc") as static:
        if surf.sizes["time"] != atmos.sizes["time"] or surf.sizes["time"] < 3:
            raise ValueError(
                "Surface and atmosphere require matching times and at least three frames"
            )
        if (surf.sizes["latitude"], surf.sizes["longitude"]) != (721, 1440):
            raise ValueError("Expected the global 0.25-degree ERA5 grid (721, 1440)")
        if tuple(atmos.level.values) != LEVELS:
            raise ValueError("ERA5 pressure levels do not match the required 13 levels")
        for ds in (atmos, static):
            for name in ("latitude", "longitude"):
                if not np.array_equal(surf[name], ds[name]):
                    raise ValueError(
                        f"ERA5 {name} coordinates differ across source files"
                    )
        if not np.array_equal(surf.time, atmos.time):
            raise ValueError("ERA5 surface and atmosphere times differ")
        if not np.all(np.diff(surf.time.values) == np.timedelta64(6, "h")):
            raise ValueError("ERA5 timestamps must be spaced six hours apart")
        expected_units = {
            "2m_temperature": "K",
            "mean_sea_level_pressure": "Pa",
            "temperature": "K",
            "specific_humidity": "kg kg**-1",
            "geopotential": "m**2 s**-2",
            "geopotential_at_surface": "m**2 s**-2",
        }
        for ds in (surf, atmos, static):
            for name in ds.data_vars:
                if name in expected_units:
                    if ds[name].attrs.get("units") != expected_units[name]:
                        raise ValueError(f"Unexpected ERA5 units for {name}")

        def tensor(values):
            values = np.asarray(values, dtype="float32")
            if not np.isfinite(values).all():
                raise ValueError("ERA5 fields must contain only finite values")
            return paddle.to_tensor(values)

        static_tensors = {
            k: tensor(static[name].transpose("latitude", "longitude").values)
            for k, name in STATIC.items()
        }

        def batch(indices):
            return Batch(
                surf_vars={
                    k: tensor(
                        surf[name]
                        .isel(time=indices)
                        .transpose("time", "latitude", "longitude")
                        .values[None]
                    )
                    for k, name in SURFACE.items()
                },
                atmos_vars={
                    k: tensor(
                        atmos[name]
                        .isel(time=indices)
                        .transpose("time", "level", "latitude", "longitude")
                        .values[None]
                    )
                    for k, name in ATMOS.items()
                },
                static_vars=static_tensors,
                metadata=Metadata(
                    lat=paddle.to_tensor(surf.latitude.values),
                    lon=paddle.to_tensor(surf.longitude.values),
                    atmos_levels=LEVELS,
                    time=(
                        surf.time.values[indices[-1]].astype("datetime64[s]").item(),
                    ),
                ),
            )

        truth_files = [
            f"truth_{index - 1:03}.nc" for index in range(2, surf.sizes["time"])
        ]
        for filename, indices in [("input.nc", [0, 1])] + [
            (filename, [index]) for index, filename in enumerate(truth_files, start=2)
        ]:
            value = batch(indices)
            temporary = root / (filename + ".partial")
            write_batch(value, temporary, args.netcdf_compression)
            temporary.replace(root / filename)
            print(f"Saved {filename}: time={value.metadata.time[0]}", flush=True)
        report = {
            "status": "ready",
            "recommended_model": "AuroraPretrained",
            "grid": [721, 1440],
            "pressure_levels_hpa": LEVELS,
            "input_times_utc": [str(t) for t in surf.time.values[:2]],
            "truth_times_utc": [str(t) for t in surf.time.values[2:]],
            "mapping": {"surface": SURFACE, "atmospheric": ATMOS, "static": STATIC},
            "normalization": "None applied here; handled by Aurora internally.",
            "input": "input.nc",
            "truth": truth_files,
        }
        (root / "batch-manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

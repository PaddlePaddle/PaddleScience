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

"""Extract a few ERA5 forecasts from WeatherBench2 without a full download."""

import argparse
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import zarr
from presets import LEVELS
from presets import PRESSURE_VARIABLES
from presets import SURFACE_VARIABLES
from presets import VARIABLES

ERA5_URL = "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


def extract(store_path, init_time, samples, stride):
    store = zarr.open_consolidated(fsspec.get_mapper(store_path), mode="r")
    raw_time = np.asarray(store["time"])
    if np.issubdtype(raw_time.dtype, np.integer):
        units, origin = store["time"].attrs["units"].split(" since ", 1)
        unit = {"hours": "h", "days": "D", "seconds": "s", "minutes": "m"}[units]
        times = pd.to_datetime(raw_time, unit=unit, origin=origin)
    else:
        times = pd.to_datetime(raw_time)
    initial = pd.Timestamp(init_time)
    index = times.searchsorted(initial)
    if index >= len(times) or times[index] != initial:
        raise ValueError("Initialization timestamp not present in store")
    indices = [index + (i - 1) * stride for i in range(samples + 2)]
    if indices[0] < 0 or indices[-1] >= len(times):
        raise ValueError("Requested history/targets lie outside store")
    for a, b in zip(indices[:-1], indices[1:]):
        if times[b] - times[a] != pd.Timedelta(hours=6 * stride):
            raise ValueError("Unexpected temporal spacing")
    longitude, latitude = np.asarray(store["longitude"]), np.asarray(store["latitude"])
    flip = latitude[0] > latitude[-1]
    if flip:
        latitude = latitude[::-1].copy()
    levels = np.asarray(store["level"]).tolist()
    fields = np.empty(
        (len(indices), len(longitude), len(latitude), len(VARIABLES)), dtype="float32"
    )

    def lon_lat(array, dims):
        if dims[-2:] == ["latitude", "longitude"]:
            array = array.T
        elif dims[-2:] != ["longitude", "latitude"]:
            raise ValueError(f"Unexpected spatial dimensions {dims}")
        return array[:, ::-1] if flip else array

    for t, source_index in enumerate(indices):
        channel = 0
        for name in SURFACE_VARIABLES:
            arr = store[name]
            fields[t, ..., channel] = lon_lat(
                np.asarray(arr[source_index]), list(arr.attrs["_ARRAY_DIMENSIONS"])
            )
            channel += 1
        for name in PRESSURE_VARIABLES:
            arr = store[name]
            dims = list(arr.attrs["_ARRAY_DIMENSIONS"])
            if dims[:2] != ["time", "level"]:
                raise ValueError(f"Unexpected pressure dimensions {dims}")
            values = np.asarray(arr[source_index])
            for level in LEVELS:
                fields[t, ..., channel] = lon_lat(values[levels.index(level)], dims)
                channel += 1
        print(f"Loaded {times[source_index]}", flush=True)
    states = np.stack([fields[i : i + 2] for i in range(samples)])
    targets = fields[2:]
    conditioning_times = times[[index + i * stride for i in range(samples)]]
    phase = np.asarray(
        [
            [
                (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400,
                (ts.dayofyear - 1) / (366 if ts.is_leap_year else 365),
            ]
            for ts in conditioning_times
        ],
        dtype="float32",
    )
    if not np.isfinite(fields).all():
        raise ValueError("Non-finite values in source weather data")
    return dict(
        state=states,
        target=targets,
        time=phase,
        longitude=longitude.astype("float32"),
        latitude=latitude.astype("float32"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=ERA5_URL)
    parser.add_argument("--init-time", default="2020-01-02T00:00")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    data = extract(args.store, args.init_time, args.samples, stride=4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source=args.store,
                init_time=args.init_time,
                samples=args.samples,
                variant="era5",
                normalized=False,
                variables=VARIABLES,
                purpose="small real-data validation slice, not a full evaluation dataset",
            ),
            indent=2,
        )
    )
    print({name: value.shape for name, value in data.items()})

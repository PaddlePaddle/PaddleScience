"""Evaluate forecasts with latitude-weighted RMSE and a persistence baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

UNITS = {
    "2t": "K",
    "10u": "m/s",
    "10v": "m/s",
    "msl": "Pa",
    "t": "K",
    "u": "m/s",
    "v": "m/s",
    "q": "kg/kg",
    "z": "m2/s2",
}


def weighted_rmse(actual, expected, latitude):
    actual = np.asarray(actual, dtype="float64")
    expected = np.asarray(expected, dtype="float64")
    latitude = np.asarray(latitude, dtype="float64")
    if (
        actual.ndim != 2
        or actual.shape != expected.shape
        or latitude.ndim != 1
        or actual.shape[0] != len(latitude)
    ):
        raise ValueError("Metric requires matching latitude x longitude arrays")
    if not all(np.isfinite(value).all() for value in (actual, expected, latitude)):
        raise ValueError("Non-finite meteorological field or coordinates")
    if np.any(np.abs(latitude) > 90):
        raise ValueError("Latitude must be in [-90, 90]")
    weights = np.maximum(np.cos(np.deg2rad(latitude)), 0)
    if weights.sum() <= 0:
        raise ValueError("Latitude weights must have a positive sum")
    return float(
        np.sqrt(
            np.sum((actual - expected) ** 2 * weights[:, None])
            / (weights.sum() * actual.shape[1])
        )
    )


def align_grid(dataset, forecast, check_time=True):
    selected = dataset.sel(latitude=forecast.latitude, longitude=forecast.longitude)
    coordinates = (
        ("latitude", "longitude", "time") if check_time else ("latitude", "longitude")
    )
    for name in coordinates:
        if not np.array_equal(selected[name], forecast[name]):
            raise ValueError(f"Forecast {name} coordinates differ from data")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    rows = []
    with xr.open_dataset(args.data / "input.nc") as initial:
        expected_names = sorted(
            name for name in initial.data_vars if name.startswith(("surf_", "atmos_"))
        )
        for step in range(1, args.steps + 1):
            with xr.open_dataset(
                args.prediction / f"prediction_{step:03}.nc"
            ) as forecast, xr.open_dataset(args.data / f"truth_{step:03}.nc") as raw:
                truth = align_grid(raw, forecast)
                persistence = align_grid(initial, forecast, check_time=False)
                if not np.array_equal(
                    forecast.level, truth.level
                ) or not np.array_equal(forecast.level, initial.level):
                    raise ValueError("Forecast and truth pressure levels differ")
                names = sorted(
                    name
                    for name in forecast.data_vars
                    if name.startswith(("surf_", "atmos_"))
                )
                if names != expected_names or any(name not in truth for name in names):
                    raise ValueError("Forecast and data variables differ")
                if forecast.sizes["batch"] != 1 or forecast.sizes["history"] != 1:
                    raise ValueError("Evaluation requires a single forecast frame")
                lead_hours = float(
                    (forecast.time.values[0] - initial.time.values[0])
                    / np.timedelta64(1, "h")
                )
                if lead_hours != step * 6:
                    raise ValueError("Expected six-hourly forecast times")
                for name in names:
                    kind, variable = name.split("_", 1)
                    levels = (
                        [int(value) for value in forecast.level.values]
                        if kind == "atmos"
                        else [None]
                    )
                    for level in levels:

                        def field(dataset, history):
                            value = dataset[name].isel(batch=0, history=history)
                            return (
                                value.sel(level=level) if level is not None else value
                            ).values

                        actual, target, held = (
                            field(forecast, 0),
                            field(truth, 0),
                            field(persistence, -1),
                        )
                        rmse = weighted_rmse(actual, target, forecast.latitude)
                        baseline = weighted_rmse(held, target, forecast.latitude)
                        rows.append(
                            {
                                "lead_hours": lead_hours,
                                "forecast_time_utc": str(forecast.time.values[0]),
                                "variable": variable,
                                "kind": kind,
                                "level_hpa": level,
                                "unit": UNITS[variable],
                                "weighted_rmse": rmse,
                                "persistence_weighted_rmse": baseline,
                                "skill_vs_persistence": 1 - rmse / baseline
                                if baseline > 0
                                else None,
                            }
                        )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "data": str(args.data),
        "prediction": str(args.prediction),
        "metric": "Cosine-latitude-weighted RMSE in native physical units; float64 accumulation",
        "scope": "One initialization; no climatological ACC or statistical benchmark",
        "forecast_metrics": rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# ERA5 forecast evaluation",
        "",
        report["metric"],
        "",
        "| Lead | Variable | Level hPa | Unit | Forecast RMSE | Persistence RMSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["kind"] == "surf" or (row["variable"], row["level_hpa"]) in (
            ("t", 850),
            ("z", 500),
        ):
            lines.append(
                f'| +{row["lead_hours"]:g}h | {row["variable"]} | {row["level_hpa"] or "surface"} | {row["unit"]} | {row["weighted_rmse"]:.6g} | {row["persistence_weighted_rmse"]:.6g} |'
            )
    lines.extend(["", report["scope"]])
    (args.output / "METRICS.md").write_text("\n".join(lines) + "\n")
    print(f"Evaluated {len(rows)} variable/level/lead combinations: {args.output}")


if __name__ == "__main__":
    main()

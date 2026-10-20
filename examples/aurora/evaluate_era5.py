"""Evaluate full-grid forecasts against ERA5 and compare Torch with Paddle.

Latitude-weighted RMSE uses cosine latitude weights, independently per variable
and pressure level. Persistence keeps the final input state unchanged. This is
a single-initialization smoke evaluation, not a multi-date benchmark or ACC.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ppsci.arch.aurora.normalisation import level_to_str
from ppsci.arch.aurora.normalisation import locations
from ppsci.arch.aurora.normalisation import scales


def weighted_rmse(actual, expected, latitude):
    actual = np.asarray(actual, dtype="float64")
    expected = np.asarray(expected, dtype="float64")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitude, dtype="float64"))), 0)
    if (
        actual.ndim != 2
        or actual.shape != expected.shape
        or actual.shape[0] != len(latitude)
    ):
        raise ValueError("Metric requires matching latitude x longitude arrays")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite meteorological field")
    return float(
        np.sqrt(
            np.sum((actual - expected) ** 2 * weights[:, None])
            / (weights.sum() * actual.shape[1])
        )
    )


def align_grid(dataset, forecast, check_time=True):
    # Select the forecast grid explicitly: Aurora commonly crops 721 to 720
    # latitudes. Never compare arrays by position without coordinate checks.
    selected = dataset.sel(latitude=forecast.latitude, longitude=forecast.longitude)
    np.testing.assert_array_equal(selected.latitude, forecast.latitude)
    np.testing.assert_array_equal(selected.longitude, forecast.longitude)
    if check_time:
        np.testing.assert_array_equal(selected.time, forecast.time)
    return selected


def assess_acceptance(
    comparisons, rows, max_abs=1e-4, relative_l2=1e-5, metric_rtol=0.01
):
    """Gate identical-input forwards and every independent weather metric."""
    first = [x for x in comparisons if x["step"] == 1]
    if not first or not rows:
        raise ValueError(
            "Acceptance requires first-step predictions and weather metrics"
        )
    forward_passed = all(
        np.isfinite(x["max_abs_normalized"])
        and np.isfinite(x["relative_l2_normalized"])
        and x["max_abs_normalized"] <= max_abs
        and x["relative_l2_normalized"] <= relative_l2
        for x in first
    )
    differences = []
    for row in rows:
        reference, actual = row["torch_weighted_rmse"], row["paddle_weighted_rmse"]
        if not np.isfinite([reference, actual]).all() or min(reference, actual) < 0:
            raise ValueError("RMSE must be finite and nonnegative")
        difference = (
            abs(actual - reference) / reference
            if reference > 0
            else (0.0 if actual == 0 else None)
        )
        row["rmse_relative_difference"] = difference
        row["metric_alignment_passed"] = (
            difference is not None and difference <= metric_rtol
        )
        differences.append(difference)
    metrics_passed = all(x["metric_alignment_passed"] for x in rows)
    return {
        "status": "passed" if forward_passed and metrics_passed else "failed",
        "same_input_first_forward": "passed" if forward_passed else "failed",
        "all_lead_rmse_alignment": "passed" if metrics_passed else "failed",
        "thresholds": {
            "first_forward_max_abs_normalized": max_abs,
            "first_forward_relative_l2_normalized": relative_l2,
            "per_variable_level_lead_rmse_relative_difference": metric_rtol,
        },
        "metric_rows": len(rows),
        "metric_failures": sum(not x["metric_alignment_passed"] for x in rows),
        "max_rmse_relative_difference": max(
            (x for x in differences if x is not None), default=None
        ),
        "note": "First forward has identical input. Later autoregressive inputs evolve independently; original pointwise tolerances remain reported for every lead without relaxation.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--torch", type=Path, required=True)
    p.add_argument("--paddle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--rtol", type=float, default=2e-4)
    p.add_argument("--atol", type=float, default=2e-4)
    p.add_argument("--metric-rtol", type=float, default=0.01)
    p.add_argument("--forward-max-abs", type=float, default=1e-4)
    p.add_argument("--forward-relative-l2", type=float, default=1e-5)
    p.add_argument("--enforce-acceptance", action="store_true")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, comparisons = [], []
    with xr.open_dataset(args.data / "input.nc") as initial:
        for step in range(1, args.steps + 1):
            with xr.open_dataset(
                args.torch / f"prediction_{step:03}.nc"
            ) as torch, xr.open_dataset(
                args.paddle / f"prediction_{step:03}.nc"
            ) as paddle, xr.open_dataset(
                args.data / f"truth_{step:03}.nc"
            ) as truth_source:
                truth = align_grid(truth_source, paddle)
                torch = align_grid(torch, paddle)
                persistence = align_grid(initial, paddle, check_time=False)
                np.testing.assert_array_equal(truth.level, paddle.level)
                np.testing.assert_array_equal(torch.level, paddle.level)
                names = sorted(
                    n for n in paddle.data_vars if n.startswith(("surf_", "atmos_"))
                )
                if names != sorted(
                    n for n in torch.data_vars if n.startswith(("surf_", "atmos_"))
                ):
                    raise ValueError("Forecast variables differ")
                expected_names = sorted(
                    n for n in initial.data_vars if n.startswith(("surf_", "atmos_"))
                )
                if names != expected_names:
                    raise ValueError("Forecast is missing input weather variables")
                for name in names:
                    kind, variable = name.split("_", 1)
                    levels = (
                        [int(x) for x in paddle.level.values]
                        if kind == "atmos"
                        else [None]
                    )
                    total_error2 = total_reference2 = 0.0
                    max_error = 0.0
                    fail_count = point_count = 0
                    for level in levels:
                        selection = {"batch": 0, "history": 0}
                        if level is not None:
                            pd = paddle[name].sel(level=level).isel(**selection).values
                            pt = torch[name].sel(level=level).isel(**selection).values
                            target = (
                                truth[name].sel(level=level).isel(**selection).values
                            )
                            held = (
                                persistence[name]
                                .sel(level=level)
                                .isel(batch=0, history=-1)
                                .values
                            )
                        else:
                            pd = paddle[name].isel(**selection).values
                            pt = torch[name].isel(**selection).values
                            target = truth[name].isel(**selection).values
                            held = persistence[name].isel(batch=0, history=-1).values
                        reference_rmse = weighted_rmse(pt, target, paddle.latitude)
                        paddle_rmse = weighted_rmse(pd, target, paddle.latitude)
                        persistent_rmse = weighted_rmse(held, target, paddle.latitude)
                        metric_difference = (
                            abs(paddle_rmse - reference_rmse) / reference_rmse
                            if reference_rmse > 0
                            else (0.0 if paddle_rmse == 0 else None)
                        )
                        rows.append(
                            {
                                "step": step,
                                "lead_hours": step * 6,
                                "forecast_time_utc": str(paddle.time.values[0]),
                                "variable": variable,
                                "kind": kind,
                                "level_hpa": level,
                                "torch_weighted_rmse": reference_rmse,
                                "paddle_weighted_rmse": paddle_rmse,
                                "persistence_weighted_rmse": persistent_rmse,
                                "rmse_relative_difference": metric_difference,
                                "metric_alignment_passed": metric_difference is not None
                                and metric_difference <= args.metric_rtol,
                                "paddle_skill_vs_persistence": 1
                                - paddle_rmse / persistent_rmse
                                if persistent_rmse > 0
                                else None,
                                "framework_weighted_rmse": weighted_rmse(
                                    pd, pt, paddle.latitude
                                ),
                            }
                        )
                        stat = (
                            variable
                            if level is None
                            else f"{variable}_{level_to_str(level)}"
                        )
                        ref = (pt.astype("float64") - locations[stat]) / scales[stat]
                        error = (pd.astype("float64") - pt.astype("float64")) / scales[
                            stat
                        ]
                        total_error2 += float(np.sum(error**2))
                        total_reference2 += float(np.sum(ref**2))
                        max_error = max(max_error, float(np.abs(error).max()))
                        fail_count += int(
                            np.count_nonzero(
                                np.abs(error) > args.atol + args.rtol * np.abs(ref)
                            )
                        )
                        point_count += error.size
                    comparisons.append(
                        {
                            "step": step,
                            "variable": name,
                            "max_abs_normalized": max_error,
                            "relative_l2_normalized": float(
                                np.sqrt(total_error2 / max(total_reference2, 1e-24))
                            ),
                            "pointwise_failures": fail_count,
                            "points": point_count,
                            "strict_pointwise_status": "passed"
                            if fail_count == 0
                            else "failed",
                        }
                    )
    acceptance = assess_acceptance(
        comparisons,
        rows,
        args.forward_max_abs,
        args.forward_relative_l2,
        args.metric_rtol,
    )
    report = {
        "status": "evaluated",
        "description": __doc__,
        "data": str(args.data),
        "normalization_metric_dtype": "float64",
        "tolerances": {"rtol": args.rtol, "atol_normalized": args.atol},
        "strict_prediction_alignment": "passed"
        if all(x["pointwise_failures"] == 0 for x in comparisons)
        else "failed",
        "alignment": comparisons,
        "forecast_metrics": rows,
        "acceptance": acceptance,
        "limits": [
            "One forecast initialization; no statistical benchmark claim",
            "No ACC: climatology not downloaded",
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    selected = [
        r
        for r in rows
        if r["kind"] == "surf"
        or (r["variable"], r["level_hpa"]) in (("t", 850), ("z", 500))
    ]
    lines = [
        "# ERA5 single-initialization evaluation",
        "",
        f"Input: {args.data}. Latitude-weighted RMSE. Native variable units.",
        "",
        "| Lead | Variable | Level hPa | Torch RMSE | Paddle RMSE | Persistence RMSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in selected:
        lines.append(
            f'| +{row["lead_hours"]}h | {row["variable"]} | {row["level_hpa"] or "surface"} | {row["torch_weighted_rmse"]:.6g} | {row["paddle_weighted_rmse"]:.6g} | {row["persistence_weighted_rmse"]:.6g} |'
        )
    lines += [
        "",
        "Strict normalized pointwise alignment: "
        + report["strict_prediction_alignment"],
        "",
        "A single forecast case does not establish general forecast skill. ACC was not computed.",
    ]
    (args.output / "METRICS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "strict_prediction_alignment": report["strict_prediction_alignment"],
                "fields_evaluated": len(rows),
            },
            indent=2,
        )
    )
    if args.enforce_acceptance and report["acceptance"]["status"] != "passed":
        raise SystemExit("ERA5 acceptance failed; inspect metrics.json")


if __name__ == "__main__":
    main()

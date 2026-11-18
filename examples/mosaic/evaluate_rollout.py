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

"""Per-lead physical weather metrics for one independent autoregressive rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_weather import weighted_mse
from presets import VARIABLES
from rollout import validate_era5_sequence


def main(args):
    if args.max_regression_percent < 0:
        raise ValueError("Regression tolerance must be nonnegative")
    actual_meta = json.loads((args.actual / "inference.json").read_text())
    reference_meta = json.loads((args.reference / "reference.json").read_text())
    for key in ("data_path", "sample_index", "seed", "amp_dtype", "rollout_steps"):
        if key not in actual_meta or actual_meta[key] != reference_meta.get(key):
            raise ValueError(f"Rollout metadata mismatch: {key}")
    if actual_meta["data_path"] != str(args.data.resolve()):
        raise ValueError("Data path mismatch")
    if actual_meta.get("reference_routes") or actual_meta.get("reference_geometry"):
        raise ValueError("Reference overrides invalidate independent evaluation")
    data, stats = np.load(args.data), np.load(args.stats)
    mean, std = [stats[name].astype("float32") for name in ("state_mean", "state_std")]
    actual = (
        np.load(args.actual / "prediction.npz")["normalized"].astype("float64") * std
        + mean
    )
    reference = (
        np.load(args.reference / "reference.npz")["normalized"].astype("float64") * std
        + mean
    )
    index, steps = actual_meta["sample_index"], actual_meta["rollout_steps"]
    validate_era5_sequence(data, index, steps)
    if (
        index < 0
        or index + steps > len(data["target"])
        or actual.shape != reference.shape
        or actual.shape[1] != steps
    ):
        raise ValueError("Invalid rollout length/targets")
    persistence = data["state"][index : index + 1, -1][:, None]
    leads = []
    for step in range(steps):
        target = data["target"][index + step : index + step + 1, None]
        values = {
            name: np.sqrt(weighted_mse(prediction, expected, data["latitude"]))
            for name, prediction, expected in (
                ("actual", actual[:, step : step + 1], target),
                ("reference", reference[:, step : step + 1], target),
                ("persistence", persistence, target),
                (
                    "cross_framework",
                    actual[:, step : step + 1],
                    reference[:, step : step + 1],
                ),
            )
        }
        if np.any(values["reference"] == 0):
            raise ValueError("Zero reference error")
        regression = (values["actual"] / values["reference"] - 1) * 100
        leads.append(
            dict(
                hours=(step + 1) * 24,
                worst_regression_percent=float(regression.max()),
                all_variables_within_tolerance=bool(
                    np.all(regression <= args.max_regression_percent)
                ),
                variables_over_tolerance=[
                    name
                    for name, value in zip(VARIABLES, regression)
                    if value > args.max_regression_percent
                ],
                variables={
                    name: dict(
                        **{
                            key + "_rmse": float(value[i])
                            for key, value in values.items()
                        },
                        regression_percent=float(regression[i]),
                    )
                    for i, name in enumerate(VARIABLES)
                },
            )
        )
    report = dict(
        scope="single-initialization autoregressive ERA5 rollout; no target feedback; internal screening only",
        actual_metadata=actual_meta,
        reference_metadata=reference_meta,
        seed=actual_meta["seed"],
        tolerance_percent=args.max_regression_percent,
        leads=leads,
        all_leads_within_tolerance=all(
            lead["all_variables_within_tolerance"] for lead in leads
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                **report,
                "leads": [
                    {k: v for k, v in lead.items() if k != "variables"}
                    for lead in leads
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "stats", "actual", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-regression-percent", type=float, default=1)
    main(parser.parse_args())

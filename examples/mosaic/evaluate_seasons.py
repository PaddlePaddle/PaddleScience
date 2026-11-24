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

"""Aggregate per-date physical RMSE reports from evaluate_weather.py."""

import argparse
import json
from pathlib import Path

import numpy as np
from presets import VARIABLES


def aggregate_reports(reports, tolerance):
    if not reports or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Need reports and a finite nonnegative tolerance")
    errors = {}
    for key in ("actual", "reference", "persistence"):
        values = np.asarray(
            [
                [report["variables"][name][key + "_rmse"] for name in VARIABLES]
                for report in reports
            ],
            dtype="float64",
        )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid RMSE values")
        errors[key] = values**2
    if np.any(errors["reference"] <= 0) or np.any(errors["persistence"] <= 0):
        raise ValueError("Relative errors require nonzero baseline RMSE")
    rmse = {key: np.sqrt(value.mean(axis=0)) for key, value in errors.items()}
    regression = 100 * (rmse["actual"] / rmse["reference"] - 1)
    per_case = 100 * (np.sqrt(errors["actual"] / errors["reference"]) - 1)
    return dict(
        sample_count=len(reports),
        tolerance_percent=tolerance,
        aggregate_all_variables_within_tolerance=bool(np.all(regression <= tolerance)),
        every_case_variable_within_tolerance=bool(np.all(per_case <= tolerance)),
        worst_aggregate_regression_percent=float(regression.max()),
        worst_case_regression_percent=float(per_case.max()),
        variables={
            name: dict(
                **{key + "_rmse": float(value[i]) for key, value in rmse.items()},
                regression_percent=float(regression[i]),
                worst_case_regression_percent=float(per_case[:, i].max()),
            )
            for i, name in enumerate(VARIABLES)
        },
    )


def main(args):
    reports = [json.loads(path.read_text()) for path in args.reports]
    summary = aggregate_reports(reports, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="Maximum relative RMSE increase in percent",
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

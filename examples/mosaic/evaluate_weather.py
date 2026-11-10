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

"""Compare physical weather errors against source predictions and persistence.

One --actual/--reference directory pair per --sample-index. All pairs must use
the same data NPZ and matching explicit noise. The threshold is an internal
screening rule, not client acceptance or a statistical significance test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from presets import VARIABLES


def weighted_mse(prediction, target, latitude):
    if prediction.shape != target.shape or prediction.shape[-1] != len(VARIABLES):
        raise ValueError("Forecast/target shape or variable ordering mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite weather values")
    weights = np.cos(np.deg2rad(np.asarray(latitude, dtype="float64")))
    if (
        weights.ndim != 1
        or not np.isfinite(weights).all()
        or np.any(np.abs(latitude) > 90)
        or len(weights) != prediction.shape[-2]
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid latitude weights")
    weights /= weights.mean()
    return np.mean(
        (prediction.astype("float64") - target.astype("float64")) ** 2
        * weights.reshape([1] * (prediction.ndim - 2) + [-1, 1]),
        axis=tuple(range(prediction.ndim - 1)),
    )


def main(args):
    if not len(args.actual) == len(args.reference) == len(args.sample_index):
        raise ValueError(
            "Need one actual/reference directory pair for each sample-index"
        )
    if len(set(args.sample_index)) != len(args.sample_index):
        raise ValueError("Duplicate sample indices would bias the aggregate")
    if args.max_regression_percent < 0:
        raise ValueError("Regression tolerance must be nonnegative")
    data, stats = np.load(args.data), np.load(args.stats)
    manifest = args.data.with_suffix(".json")
    if (
        manifest.exists()
        and json.loads(manifest.read_text()).get("variables") != VARIABLES
    ):
        raise ValueError("Data manifest variable order mismatch")
    mean, std = [stats[name].astype("float32") for name in ("state_mean", "state_std")]
    errors = {
        name: [] for name in ("actual", "reference", "persistence", "cross_framework")
    }
    cases = []
    for actual_dir, reference_dir, index in zip(
        args.actual, args.reference, args.sample_index
    ):
        if not 0 <= index < len(data["state"]):
            raise ValueError("sample-index outside data")
        actual_meta = json.loads((actual_dir / "inference.json").read_text())
        reference_meta = json.loads((reference_dir / "reference.json").read_text())
        for metadata in (actual_meta, reference_meta):
            if (
                metadata.get("data_path") != str(args.data.resolve())
                or metadata.get("sample_index") != index
            ):
                raise ValueError(
                    "Forecast metadata does not identify requested data/index; regenerate run"
                )
        if actual_meta["seed"] != reference_meta["seed"]:
            raise ValueError("Mismatched noise seeds")
        if any(
            meta.get("rollout_steps", 1) != 1 for meta in (actual_meta, reference_meta)
        ):
            raise ValueError("Use evaluate_rollout.py for multi-lead forecasts")
        if actual_meta.get("amp_dtype", "float32") != reference_meta.get(
            "amp_dtype", "float32"
        ):
            raise ValueError("Mismatched autocast precision")
        if actual_meta.get("reference_routes") or actual_meta.get("reference_geometry"):
            raise ValueError(
                "Runtime reference overrides are diagnostic, not independent evaluation"
            )
        with np.load(actual_dir / "prediction.npz") as a, np.load(
            reference_dir / "reference.npz"
        ) as b:
            actual = a["normalized"].astype("float64") * std + mean
            reference = b["normalized"].astype("float64") * std + mean
        target = data["target"][index : index + 1, None]
        persistence = data["state"][index : index + 1, -1][:, None]
        for name, prediction, expected in (
            ("actual", actual, target),
            ("reference", reference, target),
            ("persistence", persistence, target),
            ("cross_framework", actual, reference),
        ):
            errors[name].append(weighted_mse(prediction, expected, data["latitude"]))
        cases.append(
            dict(
                index=index,
                actual=str(actual_dir),
                reference=str(reference_dir),
                actual_backend=actual_meta.get("sparse_backend", "math"),
                actual_adapter=actual_meta.get("adapter_directory"),
                reference_backend=reference_meta["sparse_backend"],
                actual_amp=actual_meta.get("amp_dtype", "float32"),
                reference_amp=reference_meta.get("amp_dtype", "float32"),
            )
        )
    rmse = {name: np.sqrt(np.mean(value, axis=0)) for name, value in errors.items()}
    if np.any(rmse["reference"] == 0) or np.any(rmse["persistence"] == 0):
        raise ValueError("Zero baseline RMSE: relative screening is undefined")
    regression = (rmse["actual"] / rmse["reference"] - 1) * 100
    per_case = (
        np.sqrt(np.asarray(errors["actual"]) / np.asarray(errors["reference"])) - 1
    ) * 100
    report = dict(
        sample_count=len(cases),
        cases=cases,
        scope="equal-sample cosine-latitude weighted RMSE; limited diagnostic set; no climatology ACC",
        internal_screening=dict(
            tolerance_percent=args.max_regression_percent,
            aggregate_all_variables_within_tolerance=bool(
                np.all(regression <= args.max_regression_percent)
            ),
            every_sample_variable_within_tolerance=bool(
                np.all(per_case <= args.max_regression_percent)
            ),
            worst_aggregate_regression_percent=float(regression.max()),
            worst_case_regression_percent=float(per_case.max()),
            variables_over_tolerance=[
                name
                for name, value in zip(VARIABLES, regression)
                if value > args.max_regression_percent
            ],
            warning="Internal screening only; adjacent dates are not a representative evaluation set",
        ),
        variables={
            name: dict(
                **{key + "_rmse": float(values[i]) for key, values in rmse.items()},
                regression_percent=float(regression[i]),
                worst_case_regression_percent=float(per_case[:, i].max()),
                skill_vs_persistence=float(
                    1 - rmse["actual"][i] / rmse["persistence"][i]
                ),
            )
            for i, name in enumerate(VARIABLES)
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "variables"},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--actual", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-index", type=int, nargs="+", required=True)
    parser.add_argument("--max-regression-percent", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

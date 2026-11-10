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

"""Run isolated FP32 forecasts for several dates and aggregate physical RMSE.

Each --data file is produced by prepare_data.py; only its sample 0 is used.
GPU children run sequentially; the same visible GPU and seed are used throughout.
The optional LoRA adapter must have been trained without these evaluation dates.
"""

import argparse
import json
import os
import subprocess
import traceback
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
    if (
        not np.isfinite(args.tolerance)
        or args.tolerance < 0
        or args.timeout < 1
        or args.gpu < 0
    ):
        raise ValueError(
            "Require finite nonnegative tolerance/GPU index and positive timeout"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    report = dict(
        status="running",
        cases=[],
        seed=args.seed,
        scope="equal-date area-weighted RMSE screening, not annual evaluation; single seed and 24h lead; source uses native Triton sparse / Torch SDPA, not FlashAttention end-to-end",
    )

    def save():
        (args.output / "suite.json").write_text(json.dumps(report, indent=2))

    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))

    def run(label, python, script, options):
        print(f"Running {label}", flush=True)
        command = [str(python), str(scripts / script), *map(str, options)]
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=args.timeout,
            )

    save()
    try:
        dates = set()
        collected = {"base": [], "lora": []}
        for index, data in enumerate(args.data):
            manifest = json.loads(data.with_suffix(".json").read_text())
            if (
                manifest.get("variables") != VARIABLES
                or manifest.get("variant") != "era5"
            ):
                raise ValueError(
                    "Require ERA5 data manifests with known variable ordering"
                )
            date = manifest["init_time"]
            if date in dates:
                raise ValueError("Duplicate initialization date")
            dates.add(date)
            common = [
                "--data",
                data.resolve(),
                "--static",
                args.static,
                "--stats",
                args.stats,
                "--seed",
                args.seed,
                "--sample-index",
                0,
            ]
            reference = args.output / f"torch_{index}"
            run(
                f"torch_{index}",
                args.torch_python,
                "reference_forecast.py",
                [
                    *common,
                    "--source",
                    args.source,
                    "--config",
                    args.model / "config.json",
                    "--checkpoint",
                    args.checkpoint,
                    "--output",
                    reference,
                    "--device",
                    "cuda:0",
                    "--sparse-backend",
                    "native",
                ],
            )
            case = dict(init_time=date, data=str(data.resolve()))
            for mode in ("base", "lora") if args.adapter else ("base",):
                actual = args.output / f"{mode}_{index}"
                options = [
                    *common,
                    "--model",
                    args.model,
                    "--output",
                    actual,
                    "--device",
                    "gpu:0",
                ]
                if mode == "lora":
                    options += ["--adapter", args.adapter]
                run(f"{mode}_{index}", args.paddle_python, "infer.py", options)
                metrics = args.output / f"{mode}_{index}_metrics.json"
                run(
                    f"{mode}_{index}_metrics",
                    args.paddle_python,
                    "evaluate_weather.py",
                    [
                        "--data",
                        data.resolve(),
                        "--stats",
                        args.stats,
                        "--actual",
                        actual,
                        "--reference",
                        reference,
                        "--sample-index",
                        0,
                        "--max-regression-percent",
                        args.tolerance,
                        "--output",
                        metrics,
                    ],
                )
                result = json.loads(metrics.read_text())
                collected[mode].append(result)
                case[mode] = result["internal_screening"]
            report["cases"].append(case)
            save()
        report["aggregates"] = {
            mode: aggregate_reports(results, args.tolerance)
            for mode, results in collected.items()
            if results
        }
        report["status"] = (
            "passed"
            if all(
                value["every_case_variable_within_tolerance"]
                and value["aggregate_all_variables_within_tolerance"]
                for value in report["aggregates"].values()
            )
            else "screening_failed"
        )
        save()
        print(
            json.dumps(
                {"status": report["status"], "cases": report["cases"]}, indent=2
            ),
            flush=True,
        )
        if report["status"] != "passed":
            raise SystemExit(1)
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "source",
        "checkpoint",
        "static",
        "stats",
        "output",
        "paddle-python",
        "torch-python",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-child timeout in seconds"
    )
    parser.add_argument("--tolerance", type=float, default=1)
    main(parser.parse_args())

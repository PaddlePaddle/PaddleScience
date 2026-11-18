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

"""Compare full forecasts and sampled traces; distinguish route order from sets.

This produces diagnostic measurements, not a task-accuracy acceptance verdict.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def differences(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite comparison input")
    delta = actual.astype("float64") - expected.astype("float64")
    return dict(
        max_diff=float(np.abs(delta).max()),
        mean_diff=float(np.abs(delta).mean()),
        rmse=float(np.sqrt(np.mean(delta**2))),
    )


def main(args):
    with np.load(args.paddle / "prediction.npz") as a, np.load(
        args.reference / "reference.npz"
    ) as b:
        report = {"forecast": differences(a["normalized"], b["normalized"])}
    metadata = json.loads((args.paddle / "inference.json").read_text())
    report["diagnostic_overrides"] = {
        key: metadata.get(key) for key in ("reference_geometry", "reference_routes")
    }
    report["layers"] = {}
    report["routes"] = {}
    with np.load(args.paddle / "trace/trace.npz") as a, np.load(
        args.reference / "trace/trace.npz"
    ) as b:
        report["unmatched_keys"] = sorted(set(a.files) ^ set(b.files))
        for name in a.files:
            if name not in b:
                continue
            if name.startswith("route_"):
                actual, expected = a[name], b[name]
                if actual.shape != expected.shape:
                    raise ValueError(f"Routing shape mismatch: {name}")
                report["routes"][name] = dict(
                    differing_indices=int(np.count_nonzero(actual != expected)),
                    differing_sets=int(
                        np.count_nonzero(
                            np.any(
                                np.sort(actual, axis=-1) != np.sort(expected, axis=-1),
                                axis=-1,
                            )
                        )
                    ),
                    total_sets=int(np.prod(actual.shape[:-1])),
                )
            else:
                report["layers"][name] = differences(a[name], b[name])
    report[
        "scope"
    ] = "full forecast; sampled layer outputs; full buffers/routes; no acceptance verdict"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "layers"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())

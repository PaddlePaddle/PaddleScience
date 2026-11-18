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

"""Dynamic/CINN diagnostics for a structural fixture or full official model.

Reports compile/first-run separately, saves full-model predictions, and keeps
strict output checks separate from limited physical weather metric screening.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import paddle

from ppsci.arch.mosaic import Mosaic


def compiler_environment():
    """Record explicit compiler overrides; absence means the runtime default."""
    names = (
        "FLAGS_cinn_nvrtc_cubin_with_fmad",
        "FLAGS_use_fast_math",
        "FLAGS_deny_cinn_ops",
        "FLAGS_allow_cinn_ops",
        "FLAGS_enable_fuse_parallel_matmul_pass",
    )
    return {name: os.environ.get(name) for name in names}


def sync(device):
    if device.startswith("gpu"):
        paddle.device.synchronize()


def timed(model, inputs, warmup, steps, device):
    for _ in range(warmup):
        model(inputs)
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        result = model(inputs)["forecast"]
    sync(device)
    return (time.perf_counter() - start) / steps, result.numpy()


def main(args):
    if args.warmup < 1 or args.steps < 1:
        raise ValueError("warmup and steps must be positive")
    if args.fixture is not None and args.model is not None:
        raise ValueError("--fixture and --model are mutually exclusive")
    if args.model:
        return full_model(args)
    if args.fixture is None:
        raise ValueError("--fixture or --model is required")
    paddle.set_device(args.device)
    config = json.loads((args.fixture / "config.json").read_text())
    model = Mosaic(config, num_noise_samples=2)
    data = np.load(args.fixture / "inputs.npz")
    model.initialize_static_vars(
        *[paddle.to_tensor(data[key]) for key in ("static", "longitude", "latitude")]
    )
    model.initialize_interpolation(
        *[paddle.to_tensor(data[key]) for key in ("longitude", "latitude")]
    )
    linear = {
        name + ".weight"
        for name, layer in model.named_sublayers()
        if isinstance(layer, paddle.nn.Linear)
    }
    state = np.load(args.fixture / "initial.npz")
    model.set_state_dict(
        {
            key: paddle.to_tensor(value.T.copy() if key in linear else value)
            for key, value in state.items()
        }
    )
    model.eval()
    inputs = {key: paddle.to_tensor(data[key]) for key in ("state", "time", "noise")}
    report = dict(
        scope="small configuration, synthetic data, inference only",
        device=args.device,
        warmup=args.warmup,
        steps=args.steps,
        parameters=int(model.num_params),
        compiler_environment=compiler_environment(),
    )
    with paddle.no_grad():
        dynamic, reference = timed(model, inputs, args.warmup, args.steps, args.device)
        report["dynamic_seconds"] = dynamic
        specs = {
            key: paddle.static.InputSpec(value.shape, value.dtype, key)
            for key, value in inputs.items()
        }
        compiled = paddle.jit.to_static(
            model, input_spec=[specs], full_graph=True, backend="CINN"
        )
        sync(args.device)
        started = time.perf_counter()
        compiled(inputs)
        sync(args.device)
        report["compile_and_first_run_seconds"] = time.perf_counter() - started
        static, actual = timed(compiled, inputs, args.warmup, args.steps, args.device)
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite compiler comparison")
    delta = np.abs(actual - reference)
    report.update(
        static_seconds=static,
        max_diff=float(delta.max()),
        mean_diff=float(delta.mean()),
        speedup_percent=(dynamic / static - 1) * 100,
        precision_pass=bool(delta.max() <= args.atol),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["precision_pass"]:
        raise AssertionError("Compiler changed outputs beyond the requested tolerance")


def full_model(args):
    from convert_checkpoint import initialize_model

    if args.static is None or args.stats is None or args.data is None:
        raise ValueError("Full-model benchmark requires --static --stats --data")
    paddle.set_device("cpu")
    config = json.loads((args.model / "config.json").read_text())
    model = initialize_model(config, args.static)
    state = paddle.load(str(args.model / "model.pdparams"))
    if set(state) != set(model.state_dict()):
        raise ValueError("Checkpoint keys mismatch")
    model.set_state_dict(state)
    del state
    model.eval().to(device=args.device)
    paddle.set_device(args.device)
    data, stats, static = np.load(args.data), np.load(args.stats), np.load(args.static)
    if not 1 <= args.samples <= len(data["state"]):
        raise ValueError("Requested sample count outside data")
    for name in ("longitude", "latitude"):
        np.testing.assert_allclose(data[name], static[name], atol=1e-5, rtol=0)
    mean, std = [stats[name].astype("float32") for name in ("state_mean", "state_std")]
    inputs = dict(
        state=paddle.to_tensor(((data["state"][:1] - mean) / std)[:, None]),
        time=paddle.to_tensor(data["time"][:1, None]),
        noise=paddle.to_tensor(
            np.random.default_rng(2026)
            .standard_normal((1, config["noise_dim"]))
            .astype("float32")
        ),
    )
    report = dict(
        scope="full official model, real ERA5 samples, FP32 independent routing, inference only",
        sample_count=args.samples,
        model=str(args.model.resolve()),
        data=str(args.data.resolve()),
        device=args.device,
        parameters=int(model.num_params),
        warmup=args.warmup,
        steps=args.steps,
        status="started",
        paddle_version=paddle.__version__,
        compiler_environment=compiler_environment(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    save()
    try:
        with paddle.no_grad():
            dynamic, reference = timed(
                model, inputs, args.warmup, args.steps, args.device
            )
            report.update(dynamic_seconds=dynamic, status="compiling")
            save()
            # to_static may replace Layer.forward in place. Retain the Python
            # bound method so later samples do not compare static with itself.
            dynamic_forward = model.forward
            specs = {
                key: paddle.static.InputSpec(value.shape, value.dtype, key)
                for key, value in inputs.items()
            }
            compiled = paddle.jit.to_static(
                model, input_spec=[specs], full_graph=True, backend="CINN"
            )
            sync(args.device)
            started = time.perf_counter()
            compiled(inputs)
            sync(args.device)
            report["compile_and_first_run_seconds"] = time.perf_counter() - started
            save()
            static_seconds, actual = timed(
                compiled, inputs, args.warmup, args.steps, args.device
            )
        if not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError("Non-finite compiler output")
        delta = np.abs(actual.astype("float64") - reference.astype("float64"))
        passed = bool(delta.max() <= args.atol)
        from evaluate_weather import weighted_mse
        from presets import VARIABLES

        target = data["target"][:1, None]
        dynamic_rmse = np.sqrt(
            weighted_mse(
                reference.astype("float64") * std + mean, target, data["latitude"]
            )
        )
        compiled_rmse = np.sqrt(
            weighted_mse(
                actual.astype("float64") * std + mean, target, data["latitude"]
            )
        )
        if np.any(dynamic_rmse == 0):
            raise ValueError("Zero dynamic RMSE prevents task metric comparison")
        regression = (compiled_rmse / dynamic_rmse - 1) * 100
        saved_outputs = dict(dynamic=reference, compiled=actual)
        cases = [
            dict(
                index=0,
                max_diff=float(delta.max()),
                mean_diff=float(delta.mean()),
                worst_rmse_regression_percent=float(regression.max()),
            )
        ]
        dynamic_mses = [dynamic_rmse**2]
        compiled_mses = [compiled_rmse**2]
        with paddle.no_grad():
            for index in range(1, args.samples):
                sample_inputs = dict(inputs)
                sample_inputs["state"] = paddle.to_tensor(
                    ((data["state"][index : index + 1] - mean) / std)[:, None]
                )
                sample_inputs["time"] = paddle.to_tensor(
                    data["time"][index : index + 1, None]
                )
                dynamic_output = dynamic_forward(sample_inputs)["forecast"].numpy()
                static_output = compiled(sample_inputs)["forecast"].numpy()
                sample_delta = np.abs(
                    dynamic_output.astype("float64") - static_output.astype("float64")
                )
                if not np.isfinite(sample_delta).all():
                    raise ValueError("Non-finite multi-sample compiler output")
                target = data["target"][index : index + 1, None]
                dynamic_mse = weighted_mse(
                    dynamic_output.astype("float64") * std + mean,
                    target,
                    data["latitude"],
                )
                compiled_mse = weighted_mse(
                    static_output.astype("float64") * std + mean,
                    target,
                    data["latitude"],
                )
                if np.any(dynamic_mse == 0):
                    raise ValueError("Zero dynamic RMSE")
                sample_regression = (np.sqrt(compiled_mse / dynamic_mse) - 1) * 100
                cases.append(
                    dict(
                        index=index,
                        max_diff=float(sample_delta.max()),
                        mean_diff=float(sample_delta.mean()),
                        worst_rmse_regression_percent=float(sample_regression.max()),
                    )
                )
                dynamic_mses.append(dynamic_mse)
                compiled_mses.append(compiled_mse)
                saved_outputs[f"dynamic_{index}"] = dynamic_output
                saved_outputs[f"compiled_{index}"] = static_output
        aggregate_regression = (
            np.sqrt(np.mean(compiled_mses, axis=0) / np.mean(dynamic_mses, axis=0)) - 1
        ) * 100
        aggregate_dynamic_rmse = np.sqrt(np.mean(dynamic_mses, axis=0))
        aggregate_compiled_rmse = np.sqrt(np.mean(compiled_mses, axis=0))
        passed = all(case["max_diff"] <= args.atol for case in cases)
        np.savez(args.output.with_suffix(".npz"), **saved_outputs)
        report.update(
            status="complete" if passed else "failed",
            static_seconds=static_seconds,
            max_diff=float(delta.max()),
            mean_diff=float(delta.mean()),
            precision_pass=passed,
            tolerance=args.atol,
            cases=cases,
            multi_sample_worst_diff=max(case["max_diff"] for case in cases),
            speedup_percent=(dynamic / static_seconds - 1) * 100,
            task_metric_screening=dict(
                scope="limited adjacent dates; does not override strict output precision failure",
                aggregate_worst_rmse_regression_percent=float(
                    aggregate_regression.max()
                ),
                aggregate_variables={
                    name: dict(
                        dynamic_rmse=float(aggregate_dynamic_rmse[i]),
                        compiled_rmse=float(aggregate_compiled_rmse[i]),
                        regression_percent=float(aggregate_regression[i]),
                    )
                    for i, name in enumerate(VARIABLES)
                },
                every_sample_within_tolerance=all(
                    case["worst_rmse_regression_percent"] <= 1 for case in cases
                ),
                tolerance_percent=1.0,
                legacy_fields_scope="worst_rmse_regression_percent, all_variables_within_tolerance and variables below describe sample 0 only",
                worst_rmse_regression_percent=float(regression.max()),
                all_variables_within_tolerance=bool(np.all(regression <= 1)),
                variables={
                    name: dict(
                        dynamic_rmse=float(dynamic_rmse[i]),
                        compiled_rmse=float(compiled_rmse[i]),
                        regression_percent=float(regression[i]),
                    )
                    for i, name in enumerate(VARIABLES)
                },
            ),
        )
        save()
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            raise AssertionError("Full-model CINN output exceeds diagnostic tolerance")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Full-model precision samples; timing uses sample 0",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("Steps and warmup must be positive")
    main(args)

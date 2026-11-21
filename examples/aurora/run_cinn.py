"""Compare eager and CINN inference, direct training, or steady-state performance."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("inference", "train", "benchmark"), default="train"
    )
    parser.add_argument(
        "--benchmark-task", choices=("inference", "train"), default="train"
    )
    parser.add_argument("--backend", choices=("eager", "cinn"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--spatial-stride", type=int)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/aurora-cinn"))
    args = parser.parse_args()
    if not args.tiny and (not args.checkpoint or not args.input):
        parser.error("Full pretrained verification requires --checkpoint and --input")
    if min(args.train_steps, args.warmup, args.iterations, args.repeat) < 1:
        parser.error("Step counts, warmup, iterations and repeat must be positive")
    if args.spatial_stride is not None and args.spatial_stride < 1:
        parser.error("spatial-stride must be positive")
    args.task = args.benchmark_task if args.mode == "benchmark" else args.mode
    if args.task == "train" and not args.target:
        parser.error("Training requires a six-hour --target")
    return args


def backend_environment(backend):
    environment = dict(os.environ)
    environment.update(
        FLAGS_use_cinn=str(backend == "cinn").lower(),
        FLAGS_prim_all=str(backend == "cinn").lower(),
        FLAGS_new_executor_serial_run="true",
        NVIDIA_TF32_OVERRIDE="0",
        FLAGS_allocator_strategy="auto_growth",
    )
    return environment


def run_worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "result.json",
        "predictions.npz",
        "gradients.npz",
        "parameters.npz",
        "execution.paddle_trace.json",
    ):
        (args.output / filename).unlink(missing_ok=True)

    import paddle
    from finetune import normalized_mse
    from finetune import spatial_sample
    from finetune import validate_pair
    from sample import make_batch
    from sample import small_config

    from ppsci.arch.aurora import AuroraArch
    from ppsci.arch.aurora import Batch

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("CINN verification requires a CUDA build of PaddlePaddle")
    paddle.set_device("cpu")
    paddle.seed(args.seed)
    stride = args.spatial_stride or (1 if args.tiny else 45)
    batch = (
        spatial_sample(Batch.from_netcdf(args.input), stride)
        if args.input
        else make_batch()
    )
    model = AuroraArch(
        model_type="AuroraSmallPretrained" if args.tiny else "AuroraPretrained",
        model_kwargs=small_config() if args.tiny else {},
        metadata=batch.metadata,
        static_vars=batch.static_vars,
    )
    if args.checkpoint:
        model.model.load_checkpoint_local(str(args.checkpoint))
    target = None
    if args.task == "train":
        target = validate_pair(
            batch,
            spatial_sample(Batch.from_netcdf(args.target), stride),
            model.model.timestep,
            model.model.patch_size,
        )
    paddle.set_device("gpu:0")
    model.to("gpu:0")
    inputs = {f"surf_{k}": v.cuda() for k, v in batch.surf_vars.items()}
    inputs.update({f"atmos_{k}": v.cuda() for k, v in batch.atmos_vars.items()})
    labels = {}
    if target is not None:
        labels = {f"surf_{k}": v.cuda() for k, v in target.surf_vars.items()}
        labels.update({f"atmos_{k}": v.cuda() for k, v in target.atmos_vars.items()})
    if args.backend == "cinn":
        backbone = model.model.backbone
        for layers in (backbone.encoder_layers, backbone.decoder_layers):
            for index, layer in enumerate(layers):
                layers[index] = paddle.jit.to_static(
                    layer, backend="CINN", full_graph=True
                )
    if args.task == "train":
        model.train()
    else:
        model.eval()
    expression = normalized_mse(batch.metadata.atmos_levels)
    optimizer = paddle.optimizer.SGD(learning_rate=1e-6, parameters=model.parameters())
    losses, gradient_samples = [], {}

    def sample(value):
        size = int(np.prod(value.shape))
        return value.detach().reshape([-1])[:: max(size // 64, 1)][:64].numpy()

    initial = (
        {name: sample(value) for name, value in model.named_parameters()}
        if args.task == "train"
        else {}
    )

    def step(check=False):
        if args.task == "inference":
            with paddle.no_grad():
                return model(inputs), None
        outputs = model(inputs)
        loss = expression(outputs, labels)["normalized_mse"]
        loss.backward()
        if check:
            gradients = {
                name: p.grad
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            if not gradients or not bool(
                paddle.stack(
                    [paddle.isfinite(g).all() for g in gradients.values()]
                ).all()
            ):
                raise ValueError("Missing or non-finite gradients")
            if not gradient_samples:
                gradient_samples.update(
                    {name: sample(g) for name, g in gradients.items()}
                )
            losses.append(float(loss))
            print(f"Verified training update: loss={losses[-1]:.9g}", flush=True)
        optimizer.step()
        optimizer.clear_grad()
        return outputs, loss

    report = {
        "backend": args.backend,
        "mode": args.mode,
        "task": args.task,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "input": str(args.input.resolve()) if args.input else None,
        "target": str(args.target.resolve()) if args.target else None,
        "spatial_stride": stride,
        "batch_size": int(next(iter(inputs.values())).shape[0]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "paddle_version": paddle.__version__,
        "cuda_version": paddle.version.cuda(),
        "gpu": paddle.device.cuda.get_device_name(),
        "precision": "float32",
        "executor_serial_run": paddle.get_flags(["FLAGS_new_executor_serial_run"])[
            "FLAGS_new_executor_serial_run"
        ],
        "parameter_count": sum(int(np.prod(p.shape)) for p in model.parameters()),
        "input_grid": list(batch.spatial_shape),
        "effective_grid": list(batch.crop(model.model.patch_size).spatial_shape),
        "pressure_levels_hpa": [float(level) for level in batch.metadata.atmos_levels],
        "compiled_scope": "Full-graph capture per Swin stage with CINN subgraphs; feature expansion, encoder, decoder and Batch remain eager",
        "inference_calls_before_training": 0,
        "optimizer": {"name": "SGD", "learning_rate": 1e-6},
    }
    if args.mode == "benchmark":
        paddle.device.synchronize()
        start = time.perf_counter()
        step(check=args.task == "train")
        paddle.device.synchronize()
        report["first_step_ms"] = (time.perf_counter() - start) * 1000
        print(f"First {args.task} step: {report['first_step_ms']:.3f} ms", flush=True)
        for index in range(1, args.warmup):
            step(check=args.task == "train" and index < args.train_steps)
        paddle.device.synchronize()
        paddle.device.cuda.reset_max_memory_allocated()
        timings, rounds = [], []
        for repeat in range(args.repeat):
            current = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                outputs, loss = step()
                paddle.device.synchronize()
                current.append((time.perf_counter() - start) * 1000)
            timings.extend(current)
            rounds.append(float(np.median(current)))
            print(f"Round {repeat + 1} median: {rounds[-1]:.3f} ms", flush=True)
        report.update(
            warmup_steps=args.warmup,
            timed_steps=args.iterations * args.repeat,
            iterations_per_round=args.iterations,
            round_median_step_ms=rounds,
            step_ms=timings,
            median_step_ms=float(np.median(rounds)),
            mean_step_ms=float(np.mean(timings)),
            steps_per_second=1000 / float(np.mean(timings)),
            p95_step_ms=float(np.percentile(timings, 95)),
            peak_allocated_bytes=paddle.device.cuda.max_memory_allocated(),
        )
        if loss is not None:
            losses.append(float(loss))
        print(f"Steady-state median: {report['median_step_ms']:.3f} ms", flush=True)
    else:
        outputs, loss = None, None

    def save_trace(profiler):
        profiler.export(str(args.output / "execution.paddle_trace.json"), "json")

    def verify_steps():
        nonlocal outputs, loss
        for _ in range(args.train_steps if args.task == "train" else 1):
            outputs, loss = step(check=args.task == "train")

    if args.backend == "cinn":
        with paddle.profiler.Profiler(
            targets=[
                paddle.profiler.ProfilerTarget.CPU,
                paddle.profiler.ProfilerTarget.GPU,
            ],
            on_trace_ready=save_trace,
        ):
            if args.mode == "benchmark":
                step(check=args.task == "train")
            else:
                verify_steps()
            paddle.device.synchronize()
        events = json.loads((args.output / "execution.paddle_trace.json").read_text())[
            "traceEvents"
        ]
        report["cinn_operator_executions_in_trace"] = sum(
            "cinn_jit" in event.get("name", "").lower()
            and event.get("cat") == "Operator"
            for event in events
        )
        if not report["cinn_operator_executions_in_trace"]:
            raise RuntimeError("No CINN operator execution was found in the trace")
    elif args.mode == "benchmark":
        step(check=args.task == "train")
    else:
        verify_steps()
    if not np.isfinite(losses).all():
        raise ValueError("Non-finite loss")
    np.savez(
        args.output / "predictions.npz",
        **{key: value.numpy() for key, value in outputs.items()},
    )
    if initial:
        final = {name: sample(value) for name, value in model.named_parameters()}
        report["updated_parameter_samples"] = sum(
            not np.array_equal(initial[k], final[k]) for k in initial
        )
        if not report["updated_parameter_samples"]:
            raise ValueError("No sampled parameter changed during training")
        np.savez(args.output / "gradients.npz", **gradient_samples)
        np.savez(args.output / "parameters.npz", **final)
    report["training_losses"] = losses
    report["gradient_parameter_tensors"] = len(gradient_samples)
    report["optimizer_steps"] = (
        (args.warmup + args.repeat * args.iterations + 1)
        if args.mode == "benchmark" and args.task == "train"
        else (args.train_steps if args.task == "train" else 0)
    )
    report["first_step_scope"] = (
        "On-demand capture/compilation, execution and gradient checks"
        if args.task == "train"
        else "On-demand capture/compilation and inference"
    )
    report["training_loss_scope"] = (
        "Checked warmup updates, last timed step and untimed diagnostic update"
        if args.mode == "benchmark"
        else "Every training update"
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(eager, compiled):
    from ppsci.arch.aurora.normalisation import level_to_str
    from ppsci.arch.aurora.normalisation import locations
    from ppsci.arch.aurora.normalisation import scales

    reference = json.loads((eager / "result.json").read_text())
    result = json.loads((compiled / "result.json").read_text())
    errors = []
    with np.load(eager / "predictions.npz") as expected, np.load(
        compiled / "predictions.npz"
    ) as actual:
        for key in expected.files:
            kind, variable = key.split("_", 1)
            names = (
                [
                    f"{variable}_{level_to_str(v)}"
                    for v in reference["pressure_levels_hpa"]
                ]
                if kind == "atmos"
                else [variable]
            )
            scale = np.array([scales[name] for name in names], dtype="float64")
            location = np.array([locations[name] for name in names], dtype="float64")
            if kind == "atmos":
                scale, location = scale.reshape(1, 1, -1, 1, 1), location.reshape(
                    1, 1, -1, 1, 1
                )
            baseline = (expected[key].astype("float64") - location) / scale
            error = (actual[key].astype("float64") - expected[key]) / scale
            errors.append(
                {
                    "variable": key,
                    "max_abs_normalized": float(np.abs(error).max()),
                    "relative_l2_normalized": float(
                        np.linalg.norm(error.ravel())
                        / max(np.linalg.norm(baseline.ravel()), 1e-12)
                    ),
                }
            )
    passed = all(
        row["max_abs_normalized"] <= 1e-4 and row["relative_l2_normalized"] <= 1e-5
        for row in errors
    )
    training = {}
    if reference["task"] == "train":
        expected_losses, actual_losses = (
            reference["training_losses"],
            result["training_losses"],
        )
        matched = len(expected_losses) == len(actual_losses) and bool(
            np.allclose(expected_losses, actual_losses, rtol=1e-5, atol=1e-6)
        )
        training = {
            "losses_match": matched,
            "eager_losses": expected_losses,
            "cinn_losses": actual_losses,
        }
        passed = passed and matched
        for name in ("gradients", "parameters"):
            if not (eager / f"{name}.npz").exists():
                continue
            with np.load(eager / f"{name}.npz") as left, np.load(
                compiled / f"{name}.npz"
            ) as right:
                expected = np.concatenate([left[k].ravel() for k in left.files]).astype(
                    "float64"
                )
                actual = np.concatenate([right[k].ravel() for k in left.files]).astype(
                    "float64"
                )
            relative = float(
                np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
            )
            training[f"sampled_{name}_relative_l2"] = relative
            passed = passed and np.isfinite(relative) and relative <= 1e-4
    return {"passed": bool(passed), "forecast_errors": errors, "training": training}


def main():
    args = parse_args()
    if args.backend:
        os.environ.update(backend_environment(args.backend))
        run_worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cinn.json").unlink(missing_ok=True)
    directories = {
        backend: args.output / "run-0" / backend for backend in ("eager", "cinn")
    }
    for backend, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--backend",
            backend,
            "--output",
            str(directory),
        ]
        print(f"Running {args.mode}/{args.task}, backend {backend}", flush=True)
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(
                command,
                env=backend_environment(backend),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"{backend} failed; inspect {directory / 'run.log'}")
    comparisons = [compare(directories["eager"], directories["cinn"])]
    runs = [
        {
            backend: json.loads((path / "result.json").read_text())
            for backend, path in directories.items()
        }
    ]
    report = {
        "status": "passed" if all(row["passed"] for row in comparisons) else "failed",
        "mode": args.mode,
        "task": args.task,
        "comparisons": comparisons,
        "runs": runs,
    }
    if args.mode == "benchmark":
        medians = {
            backend: float(np.median([run[backend]["median_step_ms"] for run in runs]))
            for backend in ("eager", "cinn")
        }
        report["performance"] = {
            "median_step_ms": medians,
            "speedup": medians["eager"] / medians["cinn"],
            "rounds_per_process": args.repeat,
            "timing_scope": "Resident-input end-to-end step; training includes forward, loss, backward, SGD and gradient clearing; compilation, warmup, diagnostics, profiler and I/O excluded",
        }
    (args.output / "cinn.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], **report.get("performance", {})}, indent=2
        )
    )
    if report["status"] != "passed":
        raise SystemExit("Eager/CINN comparison failed; inspect cinn.json")


if __name__ == "__main__":
    main()

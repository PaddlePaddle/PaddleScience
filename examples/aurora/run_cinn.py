"""Verify CINN backbone execution, forecast accuracy and finite training updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import paddle
from finetune import normalized_mse
from finetune import spatial_sample
from finetune import validate_pair
from sample import make_batch
from sample import small_config

from ppsci.arch.aurora import AuroraArch
from ppsci.arch.aurora import Batch
from ppsci.arch.aurora.normalisation import level_to_str
from ppsci.arch.aurora.normalisation import locations
from ppsci.arch.aurora.normalisation import scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--spatial-stride", type=int)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/aurora-cinn"))
    args = parser.parse_args()
    if args.train_steps < 0 or args.train_steps == 1:
        parser.error("train-steps must be zero (inference only) or at least two")
    if not args.tiny and (not args.checkpoint or not args.input):
        parser.error("Full pretrained verification requires --checkpoint and --input")
    if args.train_steps and not args.target:
        parser.error("Training verification requires a six-hour --target")
    if not paddle.is_compiled_with_cuda():
        parser.error("CINN verification requires PaddlePaddle with CUDA")
    paddle.set_device("cpu")
    paddle.seed(42)
    stride = (
        args.spatial_stride
        if args.spatial_stride is not None
        else (1 if args.tiny else 45)
    )
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
    if args.train_steps:
        target = spatial_sample(Batch.from_netcdf(args.target), stride)
        target = validate_pair(
            batch, target, model.model.timestep, model.model.patch_size
        )
    paddle.set_device("gpu:0")
    model.to("gpu:0")
    inputs = {f"surf_{key}": value.cuda() for key, value in batch.surf_vars.items()}
    inputs.update(
        {f"atmos_{key}": value.cuda() for key, value in batch.atmos_vars.items()}
    )
    model.eval()
    with paddle.no_grad():
        expected = {key: value.numpy() for key, value in model(inputs).items()}
    # Keep Batch metadata and NetCDF handling outside tensor graph compilation.
    model.model.backbone = paddle.jit.to_static(
        model.model.backbone, backend="CINN", full_graph=False
    )
    with paddle.no_grad():
        actual = model(inputs)
    errors = []
    for key, tensor in actual.items():
        kind, variable = key.split("_", 1)
        names = (
            [
                f"{variable}_{level_to_str(level)}"
                for level in batch.metadata.atmos_levels
            ]
            if kind == "atmos"
            else [variable]
        )
        scale = np.asarray([scales[name] for name in names], dtype="float64")
        location = np.asarray([locations[name] for name in names], dtype="float64")
        if kind == "atmos":
            scale, location = scale.reshape(1, 1, -1, 1, 1), location.reshape(
                1, 1, -1, 1, 1
            )
        reference = (expected[key].astype("float64") - location) / scale
        error = (tensor.numpy().astype("float64") - expected[key]) / scale
        maximum = float(np.abs(error).max())
        relative = float(
            np.linalg.norm(error.ravel())
            / max(np.linalg.norm(reference.ravel()), 1e-12)
        )
        errors.append(
            {
                "variable": key,
                "max_abs_normalized": maximum,
                "relative_l2_normalized": relative,
            }
        )
    labels = {}
    if target is not None:
        labels = {
            f"surf_{key}": value.cuda() for key, value in target.surf_vars.items()
        }
        labels.update(
            {f"atmos_{key}": value.cuda() for key, value in target.atmos_vars.items()}
        )
    expression = normalized_mse(batch.metadata.atmos_levels)
    optimizer = paddle.optimizer.SGD(learning_rate=1e-6, parameters=model.parameters())
    losses = []

    def train_step():
        model.train()
        loss = expression(model(inputs), labels)["normalized_mse"]
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(paddle.isfinite(value).all()) for value in gradients
        ):
            raise ValueError("Missing or non-finite gradients")
        optimizer.step()
        optimizer.clear_grad()
        value = float(loss)
        if not np.isfinite(value):
            raise ValueError("Non-finite training loss")
        losses.append(value)

    if args.train_steps:
        train_step()
    args.output.mkdir(parents=True, exist_ok=True)
    trace_files = []

    def save_trace(profiler):
        path = args.output / "execution.paddle_trace.json"
        profiler.export(str(path), "json")
        trace_files.append(path)

    with paddle.profiler.Profiler(
        targets=[
            paddle.profiler.ProfilerTarget.CPU,
            paddle.profiler.ProfilerTarget.GPU,
        ],
        on_trace_ready=save_trace,
    ) as profiler:
        if args.train_steps:
            for _ in range(args.train_steps - 1):
                train_step()
                profiler.step()
        else:
            with paddle.no_grad():
                model(inputs)
            profiler.step()
        paddle.device.synchronize()
    events = [
        event
        for path in trace_files
        for event in json.loads(path.read_text())["traceEvents"]
    ]
    cinn_executions = sum(
        "cinn_jit" in event.get("name", "").lower() and event.get("cat") == "Operator"
        for event in events
    )
    passed = cinn_executions > 0 and all(
        np.isfinite([row["max_abs_normalized"], row["relative_l2_normalized"]]).all()
        and row["max_abs_normalized"] <= 1e-4
        and row["relative_l2_normalized"] <= 1e-5
        for row in errors
    )
    report = {
        "status": "passed" if passed else "failed",
        "paddle_version": paddle.__version__,
        "cuda_version": paddle.version.cuda(),
        "gpu": paddle.device.cuda.get_device_name(),
        "precision": "float32",
        "compiled_scope": "Swin3D backbone subgraphs; encoder, decoder and Batch processing remain eager",
        "tiny": args.tiny,
        "parameter_count": sum(
            int(np.prod(parameter.shape)) for parameter in model.parameters()
        ),
        "input_grid": list(batch.spatial_shape),
        "pressure_levels_hpa": [float(level) for level in batch.metadata.atmos_levels],
        "cinn_operator_executions_in_trace": cinn_executions,
        "thresholds": {"max_abs_normalized": 1e-4, "relative_l2_normalized": 1e-5},
        "eager_comparison": errors,
        "training_steps": len(losses),
        "training_losses": losses,
        "limits": "Sampled-grid compiler verification; does not establish full-resolution compilation or speedup",
    }
    (args.output / "cinn.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(
            "CINN verification failed; inspect cinn.json and execution.paddle_trace.json"
        )


if __name__ == "__main__":
    main()

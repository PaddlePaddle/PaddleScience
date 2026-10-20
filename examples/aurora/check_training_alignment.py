"""Compare three full-parameter SGD updates in separate Torch/Paddle processes.

The full pretrained architecture uses a 17x32 ERA5 sampling for a bounded-memory
numerical check. This is not full-resolution training or a convergence benchmark.
Original official checkpoints are never modified. Torch saves every gradient and
updated parameter; Paddle compares all of them in the original Torch layout.
"""

import argparse
import dataclasses
import gc
import json
import os
import time
from pathlib import Path

import numpy as np


def compare_arrays(actual, expected):
    a = np.asarray(actual, dtype="float64")
    b = np.asarray(expected, dtype="float64")
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Mismatching shape or non-finite training tensor")
    error = a - b
    return {
        "error_squared": float(np.sum(error * error)),
        "reference_squared": float(np.sum(b * b)),
        "max_abs": float(np.abs(error).max()),
        "elements": int(a.size),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", choices=("torch", "paddle"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--loss-rtol", type=float, default=1e-4)
    p.add_argument("--gradient-rtol", type=float, default=1e-4)
    p.add_argument("--parameter-rtol", type=float, default=1e-5)
    args = p.parse_args()
    if args.iterations < 3:
        p.error("Acceptance requires at least three updates")
    os.environ["TZ"] = "UTC"
    time.tzset()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.framework == "torch":
        import torch as f
        from aurora import AuroraPretrained
        from aurora import Batch

        f.set_num_threads(4)
        model = AuroraPretrained()
    else:
        import paddle as f

        from ppsci.arch.aurora import AuroraPretrained
        from ppsci.arch.aurora import Batch

        f.set_device("cpu")
        model = AuroraPretrained()
    model.load_checkpoint_local(str(args.checkpoint))
    model = model.to(args.device)
    if args.framework == "paddle":
        f.set_device(args.device)
    model.train()
    parameters = dict(model.named_parameters())
    linear_weights = {
        name + ".weight"
        for name, layer in (
            model.named_modules()
            if args.framework == "torch"
            else model.named_sublayers()
        )
        if isinstance(layer, f.nn.Linear)
    }

    def tensor_array(name, tensor):
        array = tensor.detach().cpu().numpy()
        return (
            array.T if args.framework == "paddle" and name in linear_weights else array
        )

    def sample(batch):
        # Global coverage, all 13 pressure levels; this intentionally changes
        # spatial sampling only for the optimizer numerical check.
        return dataclasses.replace(
            batch,
            surf_vars={k: v[..., ::45, ::45] for k, v in batch.surf_vars.items()},
            atmos_vars={k: v[..., ::45, ::45] for k, v in batch.atmos_vars.items()},
            static_vars={k: v[::45, ::45] for k, v in batch.static_vars.items()},
            metadata=dataclasses.replace(
                batch.metadata,
                lat=batch.metadata.lat[::45],
                lon=batch.metadata.lon[::45],
            ),
        ).to(args.device)

    batch = sample(Batch.from_netcdf(args.input))
    target = sample(Batch.from_netcdf(args.target)).crop(model.patch_size)
    target = target.normalise(surf_stats=model.surf_stats)
    if args.framework == "torch":
        optimizer = f.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0)
    else:
        optimizer = f.optimizer.SGD(
            learning_rate=args.learning_rate, parameters=model.parameters()
        )
    report = {
        "framework": args.framework,
        "status": "running",
        "model": "AuroraPretrained",
        "parameter_count": sum(int(np.prod(x.shape)) for x in parameters.values()),
        "input_grid": list(batch.spatial_shape),
        "levels": [int(x) for x in batch.metadata.atmos_levels],
        "precision": "float32; TF32 disabled; no autocast",
        "optimizer": {
            "type": "SGD",
            "learning_rate": args.learning_rate,
            "momentum": 0,
            "weight_decay": 0,
        },
        "loss": "Mean of nine per-variable MSEs in official normalized units",
        "thresholds": {
            "loss_relative": args.loss_rtol,
            "gradient_global_relative_l2": args.gradient_rtol,
            "updated_parameter_global_relative_l2": args.parameter_rtol,
        },
        "scope": "Full pretrained parameter updates on spatially sampled ERA5; not a full training reproduction",
        "iterations": [],
    }
    report_path = args.directory / f"training-{args.framework}.json"

    def save():
        temporary = report_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    save()
    reference_report = None
    if args.framework == "paddle":
        reference_report = json.loads(
            (args.directory / "training-torch.json").read_text()
        )
        assert reference_report["status"] == "passed"
    try:
        for step in range(1, args.iterations + 1):
            started = time.perf_counter()
            if args.framework == "torch":
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.clear_grad()
            pred = model(batch).normalise(surf_stats=model.surf_stats)
            terms = [
                ((getattr(pred, kind)[name] - truth) ** 2).mean()
                for kind in ("surf_vars", "atmos_vars")
                for name, truth in getattr(target, kind).items()
            ]
            loss = f.stack(terms).mean()
            value = float(loss.item())
            if not np.isfinite(value):
                raise ValueError("Non-finite training loss")
            loss.backward()
            unused = sorted(
                name for name, param in parameters.items() if param.grad is None
            )
            active = [name for name in parameters if name not in unused]
            row = {
                "iteration": step,
                "loss": value,
                "gradient_parameter_count": len(active),
                "unused_parameters": unused,
            }
            for phase in ("gradients", "updated_parameters"):
                if phase == "updated_parameters":
                    optimizer.step()
                archive_path = args.directory / f"torch-{step:02}-{phase}.npz"
                names = active if phase == "gradients" else list(parameters)
                if args.framework == "torch":
                    arrays = {
                        name: tensor_array(
                            name,
                            parameters[name].grad
                            if phase == "gradients"
                            else parameters[name],
                        )
                        for name in names
                    }
                    with archive_path.open("wb") as stream:
                        np.savez(stream, **arrays)
                    del arrays
                    gc.collect()
                else:
                    totals = {
                        "error_squared": 0.0,
                        "reference_squared": 0.0,
                        "max_abs": 0.0,
                        "elements": 0,
                    }
                    worst = {"name": None, "max_abs": -1.0}
                    with np.load(archive_path) as reference:
                        if set(reference.files) != set(names):
                            raise ValueError(
                                f"Gradient/parameter coverage differs: {phase}"
                            )
                        for name in names:
                            actual = tensor_array(
                                name,
                                parameters[name].grad
                                if phase == "gradients"
                                else parameters[name],
                            )
                            item = compare_arrays(actual, reference[name])
                            for key in (
                                "error_squared",
                                "reference_squared",
                                "elements",
                            ):
                                totals[key] += item[key]
                            totals["max_abs"] = max(totals["max_abs"], item["max_abs"])
                            if item["max_abs"] > worst["max_abs"]:
                                worst = {"name": name, "max_abs": item["max_abs"]}
                    totals["relative_l2"] = float(
                        np.sqrt(
                            totals["error_squared"]
                            / max(totals["reference_squared"], 1e-30)
                        )
                    )
                    totals["worst_parameter"] = worst
                    row[phase] = totals
            if reference_report:
                reference_loss = reference_report["iterations"][step - 1]["loss"]
                row["loss_relative_error"] = abs(value - reference_loss) / max(
                    abs(reference_loss), 1e-30
                )
                row["passed"] = (
                    row["loss_relative_error"] <= args.loss_rtol
                    and row["gradients"]["relative_l2"] <= args.gradient_rtol
                    and row["updated_parameters"]["relative_l2"] <= args.parameter_rtol
                )
            else:
                row["passed"] = True
            row["seconds_including_reference_io"] = time.perf_counter() - started
            report["iterations"].append(row)
            save()
            print(json.dumps(row), flush=True)
            del pred, terms, loss
        report["status"] = (
            "passed" if all(x["passed"] for x in report["iterations"]) else "failed"
        )
        save()
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        save()
        raise
    if report["status"] != "passed":
        raise SystemExit("Training alignment failed; see recorded per-iteration errors")


if __name__ == "__main__":
    main()

"""Run full pretrained Aurora on an ERA5 Batch in a separate framework process."""

import argparse
import gc
import json
import os
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from netcdf_io import write_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "paddle"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--save-paddle-checkpoint", type=Path)
    parser.add_argument("--netcdf-compression", type=int, choices=range(10), default=0)
    parser.add_argument(
        "--decoder-batch-size",
        type=int,
        default=2048,
        help="Independent Perceiver grid locations per decoder batch",
    )
    parser.add_argument(
        "--memory-probe",
        action="store_true",
        help="Explicit random-weight resource diagnostic; no forecasts are saved",
    )
    args = parser.parse_args()
    if not args.checkpoint and not args.memory_probe:
        parser.error("A pretrained --checkpoint is required")
    os.environ["TZ"] = "UTC"
    time.tzset()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "framework": args.framework,
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "device": args.device,
        "precision": "float32",
        "timezone": "UTC",
        "decoder_batch_size": args.decoder_batch_size,
        "netcdf_compression_level": args.netcdf_compression,
        "memory_probe_random_weights": args.memory_probe,
        "steps": [],
    }
    report_file = args.output / "run.json"

    def save_report():
        temporary = report_file.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_file)

    save_report()
    try:
        started = time.perf_counter()
        if args.framework == "torch":
            import torch as f
            from aurora import AuroraPretrained
            from aurora import Batch
            from aurora import rollout

            f.set_num_threads(4)
            gpu = f.cuda
            synchronize = gpu.synchronize
        else:
            import paddle as f

            from ppsci.arch.aurora import AuroraPretrained
            from ppsci.arch.aurora import Batch
            from ppsci.arch.aurora import rollout

            f.set_device(args.device if args.memory_probe else "cpu")
            gpu = f.device.cuda
            synchronize = f.device.synchronize
        model = AuroraPretrained()
        if args.decoder_batch_size <= 0:
            raise ValueError("decoder-batch-size must be positive")
        if args.framework == "torch":
            # Match the Paddle execution partition without changing original
            # parameters or interactions: each Perceiver batch item is one
            # independent grid location containing every pressure level.
            for name in ("level_decoder", "level_decoder_alternate"):
                layer = getattr(model.decoder, name, None)
                if layer is None:
                    continue
                original = layer.forward

                def chunked(levels, context, original=original):
                    size = args.decoder_batch_size
                    if levels.shape[0] <= size:
                        return original(levels, context)
                    return f.cat(
                        [
                            original(levels[i : i + size], context[i : i + size])
                            for i in range(0, levels.shape[0], size)
                        ],
                        dim=0,
                    )

                layer.forward = chunked
        else:
            model.decoder.deaggregate_levels = partial(
                model.decoder.deaggregate_levels, max_batch_size=args.decoder_batch_size
            )
        report["parameter_count"] = sum(
            int(np.prod(p.shape)) for p in model.parameters()
        )
        if args.checkpoint:
            print("Loading", args.checkpoint, flush=True)
            model.load_checkpoint_local(str(args.checkpoint))
        if args.save_paddle_checkpoint:
            assert args.framework == "paddle"
            args.save_paddle_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            f.save(model.state_dict(), str(args.save_paddle_checkpoint))
        gc.collect()
        batch = Batch.from_netcdf(args.input)
        report["input_grid"] = list(batch.spatial_shape)
        report["pressure_levels_hpa"] = [int(x) for x in batch.metadata.atmos_levels]
        report["initialization_seconds"] = time.perf_counter() - started
        print("Moving model and batch to", args.device, flush=True)
        model = model.to(args.device)
        model.eval()
        if args.framework == "paddle":
            f.set_device(args.device)
        batch = batch.to(args.device)
        synchronize()
        if args.framework == "torch":
            gpu.reset_peak_memory_stats()
        else:
            gpu.reset_max_memory_allocated()
            gpu.reset_max_memory_reserved()
        save_report()
        print("Starting full-grid forward", report["input_grid"], flush=True)
        with f.no_grad():
            predictions = rollout(model, batch, args.steps)
            for step in range(1, args.steps + 1):
                synchronize()
                begun = time.perf_counter()
                pred = next(predictions)
                synchronize()
                elapsed = time.perf_counter() - begun
                fields = list(pred.surf_vars.values()) + list(pred.atmos_vars.values())
                if not all(bool(f.isfinite(x).all().item()) for x in fields):
                    raise ValueError("Non-finite forecast")
                record = {
                    "step": step,
                    "forecast_time_utc": str(pred.metadata.time[0]),
                    "inference_seconds": elapsed,
                    "grid": list(pred.spatial_shape),
                    "peak_allocated_bytes": int(gpu.max_memory_allocated()),
                    "peak_reserved_bytes": int(gpu.max_memory_reserved()),
                }
                if not args.memory_probe:
                    path = args.output / f"prediction_{step:03}.nc"
                    temporary = path.with_suffix(".nc.partial")
                    write_batch(pred.to("cpu"), temporary, args.netcdf_compression)
                    temporary.replace(path)
                    record["file"] = path.name
                report["steps"].append(record)
                save_report()
                print(json.dumps(record), flush=True)
        report["status"] = "passed"
        report["framework_version"] = f.__version__
        save_report()
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        save_report()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

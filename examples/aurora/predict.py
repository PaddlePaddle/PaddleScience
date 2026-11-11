"""Run autoregressive Aurora forecasts from native Paddle weights."""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import paddle
from netcdf_io import write_batch
from sample import make_batch
from sample import small_config

from ppsci.arch import aurora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="AuroraPretrained",
        choices=[
            name
            for name in aurora.__all__
            if name.startswith("Aurora") and name != "AuroraArch"
        ],
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/aurora"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--tiny", action="store_true", help="Reduced random-weight demonstration"
    )
    parser.add_argument("--decoder-batch-size", type=int, default=2048)
    parser.add_argument("--netcdf-compression", type=int, choices=range(10), default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.decoder_batch_size < 1:
        parser.error("steps and decoder-batch-size must be positive")
    if args.checkpoint is None and not args.tiny:
        parser.error("Provide native --checkpoint weights")
    if args.input is None and not args.tiny:
        parser.error("Provide an --input NetCDF batch")
    if args.tiny and args.model not in ("AuroraPretrained", "AuroraSmallPretrained"):
        parser.error("The tiny example supports Pretrained and SmallPretrained")
    paddle.set_device("cpu")
    model = getattr(aurora, args.model)(**(small_config() if args.tiny else {}))
    if args.checkpoint:
        model.load_checkpoint_local(str(args.checkpoint))
    model.decoder.deaggregate_levels = partial(
        model.decoder.deaggregate_levels, max_batch_size=args.decoder_batch_size
    )
    batch = aurora.Batch.from_netcdf(args.input) if args.input else make_batch()
    paddle.set_device(args.device)
    model.to(args.device)
    model.eval()
    batch = batch.to(args.device)
    use_gpu = args.device.startswith("gpu")

    def synchronize():
        if use_gpu:
            paddle.device.synchronize()

    if use_gpu:
        paddle.device.cuda.reset_max_memory_allocated()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with paddle.no_grad():
        predictions = aurora.rollout(model, batch, args.steps)
        for step in range(1, args.steps + 1):
            synchronize()
            started = time.perf_counter()
            prediction = next(predictions)
            synchronize()
            elapsed = time.perf_counter() - started
            fields = list(prediction.surf_vars.values()) + list(
                prediction.atmos_vars.values()
            )
            if not all(bool(paddle.isfinite(value).all()) for value in fields):
                raise ValueError("Non-finite forecast")
            path = args.output / f"prediction_{step:03d}.nc"
            temporary = path.with_suffix(".nc.partial")
            write_batch(prediction.to("cpu"), temporary, args.netcdf_compression)
            temporary.replace(path)
            record = {
                "step": step,
                "forecast_time_utc": str(prediction.metadata.time[0]),
                "grid": list(prediction.spatial_shape),
                "inference_seconds": elapsed,
                "file": path.name,
            }
            if use_gpu:
                record[
                    "peak_allocated_bytes"
                ] = paddle.device.cuda.max_memory_allocated()
            records.append(record)
            print(json.dumps(record), flush=True)
    report = {
        "paddle_version": paddle.__version__,
        "model": args.model,
        "tiny": args.tiny,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "input": str(args.input) if args.input else "synthetic",
        "steps": records,
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

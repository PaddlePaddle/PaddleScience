"""Run migrated Aurora on a small synthetic batch or a saved NetCDF batch."""

import argparse
from pathlib import Path

import paddle
from sample import make_batch
from sample import small_config

from ppsci.arch import aurora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="AuroraSmallPretrained",
        choices=[
            n for n in aurora.__all__ if n.startswith("Aurora") and n != "AuroraArch"
        ],
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--input", type=Path, help="Batch NetCDF; omit to use synthetic inputs"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/aurora"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Lightweight debug architecture; incompatible with published checkpoints",
    )
    parser.add_argument("--save-paddle-checkpoint", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None and not args.tiny:
        parser.error(
            "Provide --checkpoint, or use --tiny for a random-weight smoke test"
        )
    if args.input is None and args.model not in (
        "Aurora",
        "AuroraPretrained",
        "AuroraSmall",
        "AuroraSmallPretrained",
        "Aurora12hPretrained",
    ):
        parser.error("This model variant requires a complete --input NetCDF batch")
    paddle.set_device(args.device)
    model = getattr(aurora, args.model)(**(small_config() if args.tiny else {}))
    if args.checkpoint:
        model.load_checkpoint_local(str(args.checkpoint))
    if args.save_paddle_checkpoint:
        args.save_paddle_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        paddle.save(model.state_dict(), str(args.save_paddle_checkpoint))
    model.eval()
    batch = (
        aurora.Batch.from_netcdf(args.input)
        if args.input
        else make_batch(paddle, aurora.Batch, aurora.Metadata)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with paddle.no_grad():
        for step, prediction in enumerate(aurora.rollout(model, batch, args.steps), 1):
            path = args.output / f"prediction_{step:03d}.nc"
            prediction.to("cpu").to_netcdf(path)
            print(f"Saved {path}; valid time: {prediction.metadata.time}")


if __name__ == "__main__":
    main()

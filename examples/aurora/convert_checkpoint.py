"""Convert exported Torch-layout arrays to native Paddle weights, without inference."""

import argparse
from pathlib import Path

import paddle

from ppsci.arch import aurora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="AuroraPretrained",
        choices=(
            "AuroraPretrained",
            "AuroraSmallPretrained",
        ),
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.input.suffix != ".npz" or args.output.suffix != ".pdparams":
        parser.error("Expected exported .npz input and native .pdparams output")
    paddle.set_device("cpu")
    model = getattr(aurora, args.model)()
    model.load_checkpoint_local(str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(model.state_dict(), str(args.output))
    print(f"Saved native {args.model} weights: {args.output}")


if __name__ == "__main__":
    main()

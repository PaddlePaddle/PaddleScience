"""Export an official Torch checkpoint to portable arrays without importing Paddle."""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    arrays["__format__"] = np.array("aurora-pytorch-v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez(stream, **arrays)
    print(f"Exported {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()

"""Generate offline synthetic data for exercising the Solver example."""

import argparse
import dataclasses
from datetime import timedelta
from pathlib import Path

import paddle
from sample import make_batch

from ppsci.arch.aurora import Batch
from ppsci.arch.aurora import Metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/aurora-demo"))
    args = parser.parse_args()
    paddle.set_device("cpu")
    batch = make_batch(paddle, Batch, Metadata)
    next_batch = make_batch(paddle, Batch, Metadata, seed=8)
    target = dataclasses.replace(
        next_batch,
        surf_vars={k: v[:, -1:] for k, v in next_batch.surf_vars.items()},
        atmos_vars={k: v[:, -1:] for k, v in next_batch.atmos_vars.items()},
        static_vars=batch.static_vars,
        metadata=dataclasses.replace(
            batch.metadata,
            time=tuple(t + timedelta(hours=6) for t in batch.metadata.time),
        ),
    )
    args.directory.mkdir(parents=True, exist_ok=True)
    batch.to_netcdf(args.directory / "input.nc")
    target.to_netcdf(args.directory / "truth_001.nc")
    print(f"Synthetic demonstration only: {args.directory}")


if __name__ == "__main__":
    main()

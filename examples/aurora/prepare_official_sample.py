"""Convert hash-verified official Aurora fixtures to a portable Batch NetCDF.

Pickles are read only after comparison with the pinned download manifest.
Static interpolation follows microsoft/aurora tests/conftest.py exactly.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import paddle

from ppsci.arch.aurora import Batch
from ppsci.arch.aurora import Metadata
from ppsci.arch.aurora.batch import interpolate_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    paddle.set_device("cpu")
    manifest = json.loads((args.directory / "manifest.json").read_text())
    expected = {r["file"]: r["sha256"] for r in manifest["files"]}

    def load(name):
        data = (args.directory / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected[name], name
        return pickle.loads(data)

    sample = load("aurora-0.25-small-pretrained-test-input.pickle")
    static = load("aurora-0.25-static.pickle")
    output = load("aurora-0.25-small-pretrained-test-output.pickle")
    metadata = sample["metadata"]
    static = {
        k: interpolate_numpy(
            v,
            np.linspace(90, -90, v.shape[0]),
            np.linspace(0, 360, v.shape[1], endpoint=False),
            metadata["lat"],
            metadata["lon"],
        )
        for k, v in static.items()
    }
    batch = Batch(
        surf_vars={k: paddle.to_tensor(v) for k, v in sample["surf_vars"].items()},
        static_vars={k: paddle.to_tensor(v) for k, v in static.items()},
        atmos_vars={k: paddle.to_tensor(v) for k, v in sample["atmos_vars"].items()},
        metadata=Metadata(
            lat=paddle.to_tensor(metadata["lat"]),
            lon=paddle.to_tensor(metadata["lon"]),
            time=tuple(metadata["time"]),
            atmos_levels=tuple(metadata["atmos_levels"]),
        ),
    )
    batch.to_netcdf(args.directory / "input.nc")
    np.savez(
        args.directory / "official-output.npz",
        **{
            f"{kind}.{k}": v
            for kind in ("surf_vars", "atmos_vars")
            for k, v in output[kind].items()
        },
        lat=output["metadata"]["lat"],
        lon=output["metadata"]["lon"],
        levels=output["metadata"]["atmos_levels"],
        time=str(output["metadata"]["time"][0]),
    )
    report = {
        "input_shape": {k: list(v.shape) for k, v in sample["surf_vars"].items()},
        "atmospheric_shape": {
            k: list(v.shape) for k, v in sample["atmos_vars"].items()
        },
        "input_time_utc": [str(t) for t in metadata["time"]],
        "output_time_utc": [str(t) for t in output["metadata"]["time"]],
        "levels_hpa": list(metadata["atmos_levels"]),
        "source": manifest["source"],
        "revision": manifest["revision"],
        "note": "Official regression data; meteorological provenance is not specified by the fixture.",
    }
    (args.directory / "sample-summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

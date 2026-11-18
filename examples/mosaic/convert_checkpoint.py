# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export official Torch weights to NumPy, then convert in a Paddle-only process."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from presets import model_config


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(args):
    import torch

    checksum = file_sha256(args.input)
    if args.expected_sha256 and checksum != args.expected_sha256:
        raise ValueError("Source checkpoint SHA256 mismatch")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    weights = {}
    for name, value in source.items():
        key = name.removeprefix("model.")
        if key in weights:
            raise ValueError(f"Duplicate key after stripping wrapper: {name}")
        weights[key] = value.detach().float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **weights)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                sha256=checksum,
                keys=len(weights),
                epoch=checkpoint.get("epoch"),
                source_format="official Mosaic model_state_dict; wrapper prefix stripped",
            ),
            indent=2,
        )
    )
    print(f"Exported {len(weights)} keys; SHA256={checksum}")


def initialize_model(config, static_path):
    import paddle

    from ppsci.arch.mosaic import Mosaic

    model = Mosaic(config)
    static = np.load(static_path)
    longitude = paddle.to_tensor(static["longitude"].astype("float32"))
    latitude = paddle.to_tensor(static["latitude"].astype("float32"))
    values = paddle.to_tensor(static["data"].astype("float32"))
    model.initialize_static_vars(values, longitude, latitude)
    model.initialize_interpolation(longitude, latitude)
    return model


def convert(args):
    import paddle

    paddle.set_device("cpu")
    config = model_config(args.variant)
    model = initialize_model(config, args.static)
    weights = np.load(args.input)
    params = dict(model.named_parameters())
    linear_weights = {
        name + ".weight"
        for name, layer in model.named_sublayers()
        if isinstance(layer, paddle.nn.Linear)
    }
    missing = set(params) - set(weights.files)
    unexpected = set(weights.files) - set(model.state_dict())
    if missing or unexpected:
        raise ValueError(
            f"Incomplete weights: missing={missing}, unexpected={unexpected}"
        )
    state = model.state_dict()
    for name in weights.files:
        array = weights[name].T.copy() if name in linear_weights else weights[name]
        if list(array.shape) != list(state[name].shape):
            raise ValueError(
                f"Shape mismatch for {name}: {array.shape} vs {state[name].shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite weights: {name}")
        state[name] = paddle.to_tensor(array)
    buffers = set(state) - set(params)
    if args.geometry:
        geometry_file = np.load(args.geometry)
        geometry = {
            name.removeprefix("buffer."): geometry_file[name]
            for name in geometry_file.files
            if name.startswith("buffer.") or name in buffers
        }
        if set(geometry) != buffers:
            raise ValueError("Source geometry does not cover exactly all model buffers")
        for name, array in geometry.items():
            if list(array.shape) != list(state[name].shape):
                raise ValueError(f"Source geometry shape mismatch: {name}")
            if array.dtype != state[name].numpy().dtype or not np.isfinite(array).all():
                raise ValueError(f"Source geometry dtype/non-finite mismatch: {name}")
            state[name] = paddle.to_tensor(array)
    model.set_state_dict(state)
    args.output.mkdir(parents=True, exist_ok=True)
    paddle.save(model.state_dict(), str(args.output / "model.pdparams"))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    report = dict(
        variant=args.variant,
        parameter_keys=len(params),
        loaded_parameter_keys=len(params),
        parameter_count=sum(p.numel().item() for p in params.values()),
        transposed_linear_weights=len(linear_weights),
        recomputed_buffers=0 if args.geometry else len(buffers),
        source_geometry_buffers=len(buffers) if args.geometry else 0,
        geometry_sha256=file_sha256(args.geometry) if args.geometry else None,
        weights_sha256=file_sha256(args.input),
        validation="conversion only, not task evaluation",
    )
    metadata = args.input.with_suffix(".json")
    if metadata.exists():
        report["source"] = json.loads(metadata.read_text())
    (args.output / "conversion.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["export", "convert"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["era5", "hres"], default="era5")
    parser.add_argument("--static", type=Path)
    parser.add_argument(
        "--geometry",
        type=Path,
        help="Optional original named-buffer NPZ, preserving source preprocessing exactly",
    )
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.stage == "export":
        export(args)
    else:
        if args.static is None:
            parser.error("--static is required for conversion")
        convert(args)

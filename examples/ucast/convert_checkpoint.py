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

"""Convert a released U-Cast Lightning checkpoint to Paddle parameters.

The conversion deliberately has two stages so PyTorch and PaddlePaddle do not
need to be installed in the same environment::

    python convert_checkpoint.py export-pytorch ucast.ckpt ucast.safetensors
    python convert_checkpoint.py import-paddle ucast.safetensors ucast.pdparams
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_pytorch(input_path: Path, output_path: Path, use_ema: bool) -> None:
    import torch
    from safetensors.torch import save_file

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    model_names = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.") and not key.startswith("model_ema.")
    }
    if use_ema:
        dotless_to_name = {}
        for name in model_names:
            dotless = name.replace(".", "")
            if dotless in dotless_to_name:
                raise ValueError(f"Ambiguous dotless EMA parameter name: {dotless}")
            dotless_to_name[dotless] = name
        converted = {}
        for key, value in state_dict.items():
            if not key.startswith("model_ema."):
                continue
            ema_name = key.removeprefix("model_ema.")
            if ema_name in {"decay", "num_updates"}:
                continue
            if ema_name in dotless_to_name:
                converted[dotless_to_name[ema_name]] = value.detach().contiguous()
    else:
        # Lightning's source model has non-parameter resampling buffers. Keep
        # only tensors represented by EMA as that is the Paddle model key set.
        ema_dotless_names = {
            key.removeprefix("model_ema.")
            for key in state_dict
            if key.startswith("model_ema.")
        }
        converted = {
            name: value.detach().contiguous()
            for name, value in model_names.items()
            if name.replace(".", "") in ema_dotless_names
        }

    if len(converted) != 518:
        raise ValueError(f"Expected 518 parameters, found {len(converted)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    print(f"Exported {len(converted)} tensors to {output_path}")


def import_paddle(input_path: Path, output_path: Path) -> None:
    import paddle
    from safetensors import safe_open

    state_dict = {}
    with safe_open(str(input_path), framework="numpy") as file:
        for name in file:
            state_dict[name] = paddle.to_tensor(file.get_tensor(name))
    if len(state_dict) != 518:
        raise ValueError(f"Expected 518 parameters, found {len(state_dict)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(state_dict, str(output_path))
    print(f"Imported {len(state_dict)} tensors to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-pytorch")
    export_parser.add_argument("input", type=Path)
    export_parser.add_argument("output", type=Path)
    export_parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Export training weights instead of the recommended EMA weights",
    )

    import_parser = subparsers.add_parser("import-paddle")
    import_parser.add_argument("input", type=Path)
    import_parser.add_argument("output", type=Path)

    args = parser.parse_args()
    if args.command == "export-pytorch":
        export_pytorch(args.input, args.output, use_ema=not args.raw_weights)
    else:
        import_paddle(args.input, args.output)


if __name__ == "__main__":
    main()

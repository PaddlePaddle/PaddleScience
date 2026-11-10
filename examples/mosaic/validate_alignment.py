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

"""Split-environment structural alignment against the pinned Torch source.

First run --framework torch --source /path/to/maxxxzdn/mosaic, then run
--framework paddle with the same --output. The mathematical reference replaces
only FlashAttention/Triton execution with independent, dense masked Torch math.
It is not a validation of the original BF16 Triton kernel or official weights.
Both runs compare identical inputs, explicit noise, weights and three SGD steps.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def max_abs_diff(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Non-finite values in alignment comparison")
    return float(np.max(np.abs(actual - expected)))


def small_config():
    stage = dict(
        nside=2,
        dim=32,
        num_heads=4,
        block_attn_size=4,
        sparse_block_size=4,
        sparse_block_count=2,
        encoder_depth=2,
        decoder_depth=2,
        mlp_ratio=2,
        gqa_ratio=2,
    )
    bottleneck = dict(
        nside=1,
        dim=64,
        num_heads=4,
        block_attn_size=4,
        sparse_block_size=4,
        sparse_block_count=2,
        depth=1,
        mlp_ratio=2,
        gqa_ratio=2,
    )
    return dict(
        dim=32,
        num_heads=4,
        k_neighbors=3,
        qk_norm=False,
        rope=True,
        rope_theta=10000,
        sparse_every=2,
        variables=["u", "v"],
        static_variables=["orography"],
        qkv_compress_ratio=1,
        cg_stage_cfgs=[stage],
        bottleneck_cfg=bottleneck,
        num_history_steps=2,
        noise_dim=8,
    )


def install_torch_math_attention():
    import torch
    import torch.nn.functional as F

    def flash(q, k, v):
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True).transpose(1, 2)

    def selected(q, k, v, indices, block_size):
        batch, tokens, heads, _ = k.shape
        groups = q.shape[2] // heads
        # Independent dense boolean mask, used only on tiny reference inputs.
        mask = torch.zeros(
            batch,
            heads,
            tokens // block_size,
            tokens // block_size,
            dtype=torch.bool,
            device=q.device,
        )
        mask.scatter_(-1, indices.permute(0, 2, 1, 3), True)
        mask = mask.repeat_interleave(block_size, dim=2).repeat_interleave(
            block_size, dim=3
        )
        mask = mask.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            enable_gqa=True,
        )
        return out.transpose(1, 2)

    flash_module = types.ModuleType("flash_attn")
    flash_module.flash_attn_func = flash
    ops_module = types.ModuleType("ops")
    ops_module.mosaic_sparse_attn = selected
    sys.modules["flash_attn"] = flash_module
    sys.modules["ops"] = ops_module


def torch_reference(args):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(2026)
    sys.path.insert(0, str(Path(args.source).resolve()))
    install_torch_math_attention()
    from mosaic import BottleneckConfig
    from mosaic import ModelConfig
    from mosaic import StageConfig
    from mosaic import Transformer

    cfg = small_config()
    cfg["no_compression"] = args.no_compression
    if args.noise_dim is not None:
        cfg["noise_dim"] = args.noise_dim
    if args.sparse_count is not None:
        cfg["cg_stage_cfgs"][0]["sparse_block_count"] = args.sparse_count
        cfg["bottleneck_cfg"]["sparse_block_count"] = args.sparse_count
    native = dict(cfg)
    native["cg_stage_cfgs"] = [StageConfig(**s) for s in cfg["cg_stage_cfgs"]]
    native["bottleneck_cfg"] = BottleneckConfig(**cfg["bottleneck_cfg"])
    model = Transformer(ModelConfig(**native)).to(args.device)
    rng = np.random.default_rng(2026)
    data = dict(
        state=rng.normal(size=(2, 2, 2, 8, 5, 2)).astype("float32"),
        time=rng.uniform(size=(2, 2, 2)).astype("float32"),
        static=rng.normal(size=(8, 5, 1)).astype("float32"),
        longitude=np.arange(8, dtype="float32") * 45,
        latitude=np.linspace(-80, 80, 5, dtype="float32"),
        noise=rng.normal(size=(8, cfg["noise_dim"])).astype("float32"),
        target=rng.normal(size=(2, 4, 8, 5, 2)).astype("float32"),
    )
    tensors = {k: torch.tensor(v, device=args.device) for k, v in data.items()}
    model.initialize_static_vars(
        tensors["static"], tensors["longitude"], tensors["latitude"]
    )
    model.initialize_interpolation(tensors["longitude"], tensors["latitude"])
    if cfg["noise_dim"]:

        def explicit_noise(layer, batch, device, dtype):
            return layer.to_noise(tensors["noise"])

        model.noise_generator.forward = types.MethodType(
            explicit_noise, model.noise_generator
        )
    weights = {
        name: value.detach().cpu().numpy() for name, value in model.state_dict().items()
    }
    np.savez(args.output / "initial.npz", **weights)
    np.savez(args.output / "inputs.npz", **data)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    result = {}
    for step in range(3):
        optimizer.zero_grad()
        output = model(tensors["state"], tensors["time"], 2)
        loss = ((output - tensors["target"]) ** 2).mean()
        loss.backward()
        result[f"output_{step}"] = output.detach().cpu().numpy()
        result[f"loss_{step}"] = loss.detach().cpu().numpy()
        if step == 0:
            result.update(
                {
                    f"grad::{name}": p.grad.detach().cpu().numpy()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
            )
        optimizer.step()
    np.savez(args.output / "torch.npz", **result)
    print(
        json.dumps(
            {
                "reference": "original network + mathematical attention",
                "losses": [float(result[f"loss_{i}"]) for i in range(3)],
            }
        )
    )


def paddle_compare(args):
    import paddle

    from ppsci.arch.mosaic import Mosaic

    paddle.set_device(args.device)
    cfg = json.loads((args.output / "config.json").read_text())
    model = Mosaic(cfg, num_noise_samples=2)
    arrays = np.load(args.output / "inputs.npz")
    tensors = {name: paddle.to_tensor(arrays[name]) for name in arrays.files}
    model.initialize_static_vars(
        tensors["static"], tensors["longitude"], tensors["latitude"]
    )
    model.initialize_interpolation(tensors["longitude"], tensors["latitude"])
    weights = np.load(args.output / "initial.npz")
    linear_weights = {
        name + ".weight"
        for name, layer in model.named_sublayers()
        if isinstance(layer, paddle.nn.Linear)
    }
    converted = {
        name: value.T.copy() if name in linear_weights else value
        for name, value in weights.items()
    }
    missing = set(model.state_dict()) - set(converted)
    extra = set(converted) - set(model.state_dict())
    if missing or extra:
        raise AssertionError(
            f"Checkpoint keys differ: missing={missing}, extra={extra}"
        )
    # Check geometry before loading reference buffers, so a bad port cannot be hidden.
    geometry = {}
    for name, value in model.named_buffers():
        delta = max_abs_diff(value.numpy(), converted[name])
        geometry[name] = delta
        if delta > args.atol:
            raise AssertionError(f"Geometry differs: {name} {delta}")
    model.set_state_dict(
        {name: paddle.to_tensor(value) for name, value in converted.items()}
    )
    reference = np.load(args.output / "torch.npz")
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    report = {
        "geometry_max_diff": max(geometry.values()),
        "steps": [],
        "grad_max_diff": 0.0,
        "reference_backend": "mathematical attention, not native Triton",
        "config": cfg,
        "device": args.device,
    }
    inputs = {
        "state": tensors["state"],
        "time": tensors["time"],
        "noise": tensors["noise"],
    }
    for step in range(3):
        optimizer.clear_grad()
        output = model(inputs)["forecast"]
        loss = ((output - tensors["target"]) ** 2).mean()
        loss.backward()
        diff = max_abs_diff(output.numpy(), reference[f"output_{step}"])
        loss_diff = abs(float(loss) - float(reference[f"loss_{step}"]))
        report["steps"].append(
            dict(step=step, loss=float(loss), max_diff=diff, loss_diff=loss_diff)
        )
        if diff > args.atol or loss_diff > args.atol:
            raise AssertionError(f"Forward/loss mismatch: {report['steps'][-1]}")
        if step == 0:
            for name, param in model.named_parameters():
                key = "grad::" + name
                if (param.grad is None) != (key not in reference):
                    raise AssertionError(f"Gradient presence mismatch: {name}")
                if param.grad is not None:
                    grad = param.grad.numpy()
                    if name in linear_weights:
                        grad = grad.T
                    delta = max_abs_diff(grad, reference[key])
                    report["grad_max_diff"] = max(report["grad_max_diff"], delta)
                    if delta > args.atol:
                        raise AssertionError(f"Gradient mismatch: {name} {delta}")
        optimizer.step()
    report["passed"] = True
    (args.output / "alignment.json").write_text(json.dumps(report, indent=2))
    device_tag = args.device.replace(":", "_")
    (args.output / f"alignment_{device_tag}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["torch", "paddle"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--noise-dim", type=int)
    parser.add_argument("--sparse-count", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.framework == "torch":
        if args.source is None:
            parser.error("--source is required for the original Torch implementation")
        torch_reference(args)
    else:
        paddle_compare(args)

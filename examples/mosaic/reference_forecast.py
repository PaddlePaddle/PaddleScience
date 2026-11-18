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

"""Original Torch network forecast, with selectable sparse kernel reference.

The local/compressed attention uses Torch SDPA instead of FlashAttention's Python
package. --sparse-backend native runs the original Triton sparse kernel without
modification; math uses independent chunked SDPA with no intermediate BF16 cast.
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mosaic_trace import Trace
from rollout import forecast_rollout
from rollout import validate_era5_sequence


def install_attention(backend):
    def flash(q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)

    module = types.ModuleType("flash_attn")
    module.flash_attn_func = flash
    sys.modules["flash_attn"] = module
    if backend == "native":
        return

    def selected(q, k, v, indices, block_size):
        batch, length, heads, dim = k.shape
        blocks, groups = length // block_size, q.shape[2] // heads
        output = torch.empty_like(q)
        # Process each KV head independently, with a chunk of query blocks.
        for h in range(heads):
            kb = k[:, :, h].reshape(batch, blocks, block_size, dim)
            vb = v[:, :, h].reshape(batch, blocks, block_size, dim)
            for start in range(0, blocks, 8):
                stop = min(blocks, start + 8)
                choice = indices[:, start:stop, h]
                row = torch.arange(batch, device=k.device)[:, None, None]
                keys = kb[row, choice].reshape(batch * (stop - start), -1, dim)
                values = vb[row, choice].reshape_as(keys)
                queries = q[
                    :,
                    start * block_size : stop * block_size,
                    h * groups : (h + 1) * groups,
                ]
                queries = queries.reshape(
                    batch * (stop - start), block_size, groups, dim
                ).transpose(1, 2)
                result = F.scaled_dot_product_attention(
                    queries, keys[:, None], values[:, None], enable_gqa=True
                )
                result = result.transpose(1, 2).reshape(
                    batch, (stop - start) * block_size, groups, dim
                )
                output[
                    :,
                    start * block_size : stop * block_size,
                    h * groups : (h + 1) * groups,
                ] = result
        return output

    ops = types.ModuleType("ops")
    ops.mosaic_sparse_attn = selected
    sys.modules["ops"] = ops


def main(args):
    if args.rollout_steps < 1:
        raise ValueError("rollout-steps must be positive")
    if args.rollout_steps > 1 and (args.trace or args.sparse_fixture):
        raise ValueError("Rollout evaluation must not capture single-step traces")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(args.source.resolve()))
    install_attention(args.sparse_backend)
    from mosaic import BottleneckConfig
    from mosaic import ModelConfig
    from mosaic import StageConfig
    from mosaic import Transformer

    cfg = json.loads(args.config.read_text())
    cfg["cg_stage_cfgs"] = [StageConfig(**s) for s in cfg["cg_stage_cfgs"]]
    cfg["bottleneck_cfg"] = BottleneckConfig(**cfg["bottleneck_cfg"])
    model = Transformer(ModelConfig(**cfg))
    static = np.load(args.static)
    lon, lat = [
        torch.from_numpy(static[k].astype("float32")) for k in ("longitude", "latitude")
    ]
    model.initialize_static_vars(torch.from_numpy(static["data"]), lon, lat)
    model.initialize_interpolation(lon, lat)
    if args.geometry_output:
        args.geometry_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.geometry_output,
            **{
                name: value.detach().cpu().numpy()
                for name, value in model.named_buffers()
            },
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = {
        name.removeprefix("model."): value
        for name, value in checkpoint["model_state_dict"].items()
    }
    if set(dict(model.named_parameters())) - set(weights):
        raise ValueError("Official checkpoint has missing trainable parameters")
    incompat = model.load_state_dict(weights, strict=False)
    if incompat.unexpected_keys:
        raise ValueError(f"Unexpected source keys: {incompat.unexpected_keys}")
    del weights, checkpoint
    model = model.eval().to(args.device)
    data, stats = np.load(args.data), np.load(args.stats)
    index = args.sample_index
    if not 0 <= index < len(data["state"]):
        raise ValueError("sample-index outside data")
    if index + args.rollout_steps > len(data["target"]):
        raise ValueError("Insufficient targets for requested rollout")
    validate_era5_sequence(data, index, args.rollout_steps)
    mean, std = [stats[key].astype("float32") for key in ("state_mean", "state_std")]
    state = torch.tensor(
        ((data["state"][index : index + 1] - mean) / std)[:, None], device=args.device
    )
    phase = torch.tensor(data["time"][index : index + 1, None], device=args.device)
    noise = torch.tensor(
        np.random.default_rng(args.seed)
        .standard_normal((1, cfg["noise_dim"]))
        .astype("float32"),
        device=args.device,
    )

    def explicit_noise(layer, batch, device, dtype):
        return layer.to_noise(noise)

    model.noise_generator.forward = types.MethodType(
        explicit_noise, model.noise_generator
    )
    trace = Trace(args.output / "trace") if args.trace else None
    sparse_diagnostics = None
    if args.sparse_diagnostics or args.sparse_sensitivity:
        if args.sparse_backend != "native" or args.amp_dtype != "float32":
            raise ValueError("Sparse diagnostics require native FP32 reference")
        from sparse_diagnostics import install_diagnostics

        sparse_diagnostics = install_diagnostics(sensitivity=args.sparse_sensitivity)
    if args.sparse_fixture:
        import primitives

        original_sparse = primitives.mosaic_sparse_attn
        saved_sparse = []

        def capture_sparse(q, k, v, indices, block_size):
            result = original_sparse(q, k, v, indices, block_size)
            if not saved_sparse:
                args.sparse_fixture.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    args.sparse_fixture,
                    **{
                        name: tensor.detach().float().cpu().numpy()
                        for name, tensor in (
                            ("q", q),
                            ("k", k),
                            ("v", v),
                            ("reference", result),
                        )
                    },
                    indices=indices.detach().cpu().numpy(),
                    block_size=block_size,
                )
                saved_sparse.append(True)
            return result

        primitives.mosaic_sparse_attn = capture_sparse
    if trace is not None:
        trace.install_torch(model)
        import primitives

        original_topk = primitives.attn_topk
        calls = []

        def traced_topk(q, k, count):
            result = original_topk(q, k, count)
            trace.record(f"route_{len(calls)}", result, full=True)
            calls.append(None)
            return result

        primitives.attn_topk = traced_topk
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        "cuda", enabled=args.amp_dtype != "float32", dtype=torch.float16
    ):
        if args.rollout_steps == 1:
            output = model(state, phase, 1)
            normalized = output.cpu().numpy()
        else:

            def predict(history, time_phase, draw):
                nonlocal noise
                noise = torch.tensor(draw, device=args.device)
                return (
                    model(
                        torch.tensor(history[:, None], device=args.device),
                        torch.tensor(time_phase[:, None], device=args.device),
                        1,
                    )
                    .cpu()
                    .numpy()
                )

            normalized = forecast_rollout(
                predict,
                data["state"][index : index + 1],
                data["time"][index : index + 1],
                mean,
                std,
                cfg["noise_dim"],
                args.seed,
                args.rollout_steps,
            )
    torch.cuda.synchronize()
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite reference prediction")
    report = dict(
        sparse_backend=args.sparse_backend,
        local_backend="torch SDPA",
        seconds=time.perf_counter() - started,
        shape=list(normalized.shape),
        dtype=str(normalized.dtype),
        amp_dtype=args.amp_dtype,
        device=args.device,
        seed=args.seed,
        data_path=str(args.data.resolve()),
        sample_index=index,
        rollout_steps=args.rollout_steps,
    )
    if args.paddle_prediction:
        actual = np.load(args.paddle_prediction)["normalized"]
        if not np.isfinite(actual).all():
            raise ValueError("Non-finite Paddle prediction")
        delta = np.abs(actual - normalized)
        report.update(
            max_diff=float(delta.max()),
            mean_diff=float(delta.mean()),
            rmse=float(np.sqrt(np.mean(delta**2))),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if sparse_diagnostics is not None:
        (args.output / "sparse_diagnostics.json").write_text(
            json.dumps(
                dict(
                    calls=sparse_diagnostics,
                    scope="fixed native input/routes at every call; alternative outputs never feed subsequent layers; forward only, not training or speed validation",
                ),
                indent=2,
            )
        )
        report["sparse_diagnostics"] = True
    if trace is not None:
        trace.save()
    np.savez(args.output / "reference.npz", normalized=normalized)
    (args.output / "reference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "checkpoint", "static", "stats", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument(
        "--sparse-backend", choices=("native", "math"), default="native"
    )
    parser.add_argument("--paddle-prediction", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--sparse-diagnostics", action="store_true")
    parser.add_argument(
        "--sparse-sensitivity",
        action="store_true",
        help="Add fixed-route one-ULP Q perturbations to sparse diagnostics",
    )
    parser.add_argument(
        "--sparse-fixture",
        type=Path,
        help="Capture the first actual sparse call for isolated replay",
    )
    parser.add_argument(
        "--amp-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        help="Export original preprocessing/position buffers for checkpoint conversion",
    )
    main(parser.parse_args())

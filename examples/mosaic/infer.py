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

"""Full-resolution Mosaic one-step inference on an extracted physical ERA5 sample."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import paddle
from convert_checkpoint import initialize_model
from mosaic_trace import Trace
from rollout import forecast_rollout
from rollout import validate_era5_sequence


def main(args):
    if args.rollout_steps < 1:
        raise ValueError("rollout-steps must be positive")
    if args.rollout_steps > 1 and (
        args.trace or args.reference_routes or args.reference_trace
    ):
        raise ValueError("Rollout evaluation must not use trace or reference overrides")
    paddle.set_device("cpu")
    config = json.loads((args.model / "config.json").read_text())
    model = initialize_model(config, args.static)
    state = paddle.load(str(args.model / "model.pdparams"))
    if set(state) != set(model.state_dict()):
        raise ValueError(
            "Native checkpoint keys do not match initialized geometry/model"
        )
    model.set_state_dict(state)
    del state
    adapter_manifest = None
    if args.adapter:
        from lora import inject_lora
        from lora import load_adapter

        manifest = json.loads((args.adapter / "adapter_config.json").read_text())
        adapter_manifest = inject_lora(model, manifest["rank"], manifest["alpha"])
        if adapter_manifest != manifest:
            raise ValueError("LoRA manifest does not match model targets")
        load_adapter(model, args.adapter / "adapter.pdparams")
    if args.reference_trace:
        # Diagnostic only: isolate regenerated geometry from forward arithmetic.
        reference = np.load(args.reference_trace)
        for name, buffer in model.named_buffers():
            value = reference["buffer." + name]
            if list(value.shape) != list(buffer.shape):
                raise ValueError(f"Reference buffer shape mismatch: {name}")
            buffer.set_value(value)
    model.eval()
    model.to(device=args.device)
    paddle.set_device(args.device)
    if args.sparse_backend in ("native-emulation", "qkv-bf16"):
        from validate_sparse_kernel import online_sparse_attention
        from validate_sparse_kernel import quantized_qkv_attention

        from ppsci.arch import mosaic_attention

        mosaic_attention.sparse_attention = (
            online_sparse_attention
            if args.sparse_backend == "native-emulation"
            else quantized_qkv_attention
        )
    sample, stats, static = (
        np.load(args.data),
        np.load(args.stats),
        np.load(args.static),
    )
    for key in ("longitude", "latitude"):
        np.testing.assert_allclose(sample[key], static[key], rtol=0, atol=1e-5)
    mean, std = stats["state_mean"].astype("float32"), stats["state_std"].astype(
        "float32"
    )
    index = args.sample_index
    if not 0 <= index < len(sample["state"]):
        raise ValueError("sample-index outside data")
    raw = sample["state"][index : index + 1]
    validate_era5_sequence(sample, index, args.rollout_steps)
    if index + args.rollout_steps > len(sample["target"]):
        raise ValueError("Insufficient targets for requested rollout")
    if raw.shape[1] != config["num_history_steps"]:
        raise ValueError("History length does not match model variant")
    inputs = {
        "state": paddle.to_tensor(((raw - mean) / std)[:, None]),
        "time": paddle.to_tensor(sample["time"][index : index + 1, None]),
        "noise": paddle.to_tensor(
            np.random.default_rng(args.seed)
            .standard_normal((1, config["noise_dim"]))
            .astype("float32")
        ),
    }
    trace = Trace(args.output / "trace") if args.trace else None
    if trace is not None:
        trace.install_paddle(model)
    if trace is not None or args.reference_routes:
        from ppsci.arch import mosaic_attention

        original_topk = mosaic_attention.attn_topk
        calls = []
        routes = np.load(args.reference_routes) if args.reference_routes else None

        def traced_topk(q, k, count):
            result = original_topk(q, k, count)
            name = f"route_{len(calls)}"
            if routes is not None:
                value = routes[name]
                if list(value.shape) != list(result.shape):
                    raise ValueError(f"Reference routing shape mismatch: {name}")
                result = paddle.to_tensor(value, place=result.place)
            if trace is not None:
                trace.record(name, result, full=True)
            calls.append(None)
            return result

        mosaic_attention.attn_topk = traced_topk
    if args.device.startswith("gpu"):
        paddle.device.synchronize()
    start = time.perf_counter()
    with paddle.no_grad(), paddle.amp.auto_cast(
        enable=args.amp_dtype != "float32", dtype="float16", level="O1"
    ):
        if args.rollout_steps == 1:
            result = model(inputs)["forecast"]
            normalized = result.numpy()
        else:

            def predict(history, phase, noise):
                return model(
                    {
                        "state": paddle.to_tensor(history[:, None]),
                        "time": paddle.to_tensor(phase[:, None]),
                        "noise": paddle.to_tensor(noise),
                    }
                )["forecast"].numpy()

            normalized = forecast_rollout(
                predict,
                raw,
                sample["time"][index : index + 1],
                mean,
                std,
                config["noise_dim"],
                args.seed,
                args.rollout_steps,
            )
    if args.device.startswith("gpu"):
        paddle.device.synchronize()
    elapsed = time.perf_counter() - start
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite prediction")
    prediction = normalized * std + mean
    if index + args.rollout_steps > len(sample["target"]):
        raise ValueError("Insufficient targets for requested rollout")
    target = sample["target"][index : index + args.rollout_steps][None]
    latitude_weights = np.cos(np.deg2rad(sample["latitude"]))
    latitude_weights /= latitude_weights.mean()
    # Per-variable diagnostics; one sample cannot establish full weather skill.
    error = prediction - target
    rmse = np.sqrt(
        np.mean(
            error**2 * latitude_weights[None, None, None, :, None], axis=(0, 1, 2, 3)
        )
    )
    report = dict(
        device=args.device,
        model_directory=str(args.model.resolve()),
        adapter_directory=str(args.adapter.resolve()) if args.adapter else None,
        adapter_config=adapter_manifest,
        conversion=(
            json.loads((args.model / "conversion.json").read_text())
            if (args.model / "conversion.json").exists()
            else None
        ),
        seed=args.seed,
        data_path=str(args.data.resolve()),
        sample_index=index,
        rollout_steps=args.rollout_steps,
        shape=list(prediction.shape),
        elapsed_seconds=elapsed,
        parameters=int(model.num_params),
        finite=True,
        sample_count=1,
        compiler="dynamic",
        sparse_backend=args.sparse_backend,
        amp_dtype=args.amp_dtype,
        reference_geometry=str(args.reference_trace) if args.reference_trace else None,
        reference_routes=str(args.reference_routes) if args.reference_routes else None,
        variant_history=config["num_history_steps"],
        normalized_min=float(normalized.min()),
        normalized_max=float(normalized.max()),
        area_weighted_rmse={
            name: float(value) for name, value in zip(config["variables"], rmse)
        },
        scope="full-resolution real-input diagnostic; lead count in rollout_steps; not full evaluation or speed benchmark",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if trace is not None:
        trace.save()
    np.savez(
        args.output / "prediction.npz",
        forecast=prediction,
        normalized=normalized,
        longitude=sample["longitude"],
        latitude=sample["latitude"],
    )
    (args.output / "inference.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Optional LoRA adapter directory")
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument(
        "--sparse-backend",
        choices=("math", "native-emulation", "qkv-bf16"),
        default="math",
        help="native-emulation and qkv-bf16 are inference-only diagnostics, not training backends",
    )
    parser.add_argument(
        "--reference-trace", type=Path, help="Diagnostic geometry replay only"
    )
    parser.add_argument(
        "--reference-routes",
        type=Path,
        help="Diagnostic frozen routing only; not independent inference",
    )
    main(parser.parse_args())

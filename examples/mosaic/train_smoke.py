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

"""Real-input FP32 full-weight or LoRA diagnostic, not a training reproduction.

Supports reference variable/latitude weighted MSE, MAE and fair ensemble CRPS.
Checks every trainable gradient and performs SGD updates with fixed sample/noise.
It is NOT the paper's full training schedule or an independent skill test.
"""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import paddle
from convert_checkpoint import initialize_model
from lora import adapter_state
from lora import inject_lora
from lora import load_adapter
from losses import ensemble_forecast
from losses import weighted_loss
from presets import LEVELS
from presets import PRESSURE_VARIABLES
from presets import SURFACE_VARIABLES


def variable_weights(variables):
    surface = dict(zip(SURFACE_VARIABLES, (1.0, 0.1, 0.1, 0.1)))
    weights = []
    for name in variables:
        if name in surface:
            weights.append(surface[name])
        else:
            field, level = name.rsplit("_", 1)
            if field not in PRESSURE_VARIABLES or int(level) not in LEVELS:
                raise ValueError(f"Unknown weighted variable: {name}")
            weights.append(int(level) / np.mean(LEVELS))
    return np.asarray(weights, dtype="float32")


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if param.stop_gradient:
            digest.update(name.encode())
            digest.update(param.numpy().tobytes())
    return digest.hexdigest()


def main(args):
    if args.steps < 1 or not 0 < args.learning_rate < 1:
        raise ValueError("Require steps >= 1 and 0 < learning-rate < 1")
    if args.members < 1 or (args.loss == "crps" and args.members < 2):
        raise ValueError("Positive members required; fair CRPS needs >=2")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="initializing",
        steps=[],
        device=args.device,
        model=str(args.model.resolve()),
        data=str(args.data.resolve()),
        sample_index=args.sample_index,
        seed=args.seed,
        dtype="float32",
        recompute=not args.no_recompute,
        optimizer="SGD",
        learning_rate=args.learning_rate,
        training_mode="lora" if args.lora_rank else "full",
        loss_type=args.loss,
        ensemble_members=args.members,
        scope="trainability diagnostic; fixed real sample/noise; not paper training reproduction or generalization evaluation",
    )

    def save():
        (args.output / "training.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        paddle.set_device("cpu")
        config = json.loads((args.model / "config.json").read_text())
        model = initialize_model(config, args.static)
        state = paddle.load(str(args.model / "model.pdparams"))
        if set(state) != set(model.state_dict()):
            raise ValueError("Checkpoint key mismatch")
        model.set_state_dict(state)
        del state
        paddle.seed(args.seed)
        if args.lora_rank:
            report["lora"] = inject_lora(model, args.lora_rank, args.lora_alpha)
        model.set_recompute(not args.no_recompute)
        model.train()
        model.to(device=args.device)
        paddle.set_device(args.device)
        paddle.seed(args.seed)
        with np.load(args.data) as data, np.load(args.stats) as stats, np.load(
            args.static
        ) as static:
            for key in ("longitude", "latitude"):
                np.testing.assert_allclose(data[key], static[key], rtol=0, atol=1e-5)
            index = args.sample_index
            if not 0 <= index < min(len(data["state"]), len(data["target"])):
                raise ValueError("sample-index outside data")
            mean, std = stats["state_mean"], stats["state_std"]
            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("Invalid normalization standard deviations")
            raw = data["state"][index : index + 1]
            if raw.shape[1] != config["num_history_steps"]:
                raise ValueError("History length mismatch")
            inputs = dict(
                state=paddle.to_tensor(((raw - mean) / std)[:, None].astype("float32")),
                time=paddle.to_tensor(
                    data["time"][index : index + 1, None].astype("float32")
                ),
                noise=paddle.to_tensor(
                    np.random.default_rng(args.seed)
                    .standard_normal((args.members, config["noise_dim"]))
                    .astype("float32")
                ),
            )
            target = paddle.to_tensor(
                ((data["target"][index : index + 1, None] - mean) / std).astype(
                    "float32"
                )
            )
            lat = np.cos(np.deg2rad(data["latitude"].astype("float64")))
            lat /= lat.mean()
            weights = paddle.to_tensor(
                (lat[:, None] * variable_weights(config["variables"])[None]).astype(
                    "float32"
                )
            )[None, None, None]
        all_params = list(model.named_parameters())
        params = [(name, p) for name, p in all_params if not p.stop_gradient]
        report["parameters"] = sum(int(p.numel()) for _, p in all_params)
        report["trainable_parameters"] = sum(int(p.numel()) for _, p in params)
        if args.lora_rank:
            report["frozen_sha256_before"] = frozen_digest(model)
        report["parameter_tensors"] = len(params)
        optimizer = paddle.optimizer.SGD(
            learning_rate=args.learning_rate, parameters=[p for _, p in params]
        )
        report["status"] = "training"

        def forecast():
            return ensemble_forecast(model, inputs, args.members, not args.no_recompute)

        def score(prediction):
            return weighted_loss(prediction, target, weights, args.loss)

        save()
        for step in range(args.steps):
            started = time.perf_counter()
            prediction = forecast()
            loss = score(prediction)
            value = float(loss)
            if not np.isfinite(value):
                raise ValueError("Non-finite training loss")
            report["phase"] = f"backward_{step}"
            save()
            loss.backward()
            gradients = {}
            for name, param in params:
                if param.grad is None or not bool(paddle.isfinite(param.grad).all()):
                    raise ValueError(f"Missing or non-finite gradient: {name}")
                gradients[name] = float(paddle.linalg.norm(param.grad))
            norm = float(np.linalg.norm(list(gradients.values())))
            if norm == 0:
                raise ValueError("All gradients zero")
            optimizer.step()
            optimizer.clear_grad()
            for name, param in params:
                if not bool(paddle.isfinite(param).all()):
                    raise ValueError(f"Non-finite updated parameter: {name}")
            record = dict(
                step=step,
                loss=value,
                gradient_norm=norm,
                gradients=gradients,
                seconds=time.perf_counter() - started,
            )
            report["steps"].append(record)
            save()
            print(
                json.dumps({k: v for k, v in record.items() if k != "gradients"}),
                flush=True,
            )
            del prediction, loss
        with paddle.no_grad():
            after = float(score(forecast()))
        report["final_loss"] = after
        report["fixed_sample_loss_decreased"] = bool(
            np.isfinite(after) and after < report["steps"][0]["loss"]
        )
        if not report["fixed_sample_loss_decreased"]:
            raise ValueError("Fixed-sample loss did not decrease")
        if args.lora_rank:
            report["frozen_sha256_after"] = frozen_digest(model)
            if report["frozen_sha256_before"] != report["frozen_sha256_after"]:
                raise ValueError("Frozen base parameters changed")
            adapter_path = args.output / "adapter.pdparams"
            paddle.save(adapter_state(model), str(adapter_path))
            (args.output / "adapter_config.json").write_text(
                json.dumps(report["lora"], indent=2)
            )
            with paddle.no_grad():
                expected = forecast().numpy()
                for param in adapter_state(model).values():
                    param.set_value(paddle.zeros_like(param))
                load_adapter(model, adapter_path)
                restored = forecast().numpy()
            np.testing.assert_array_equal(expected, restored)
            report["adapter_roundtrip_exact"] = True
        if args.save_checkpoint:
            paddle.save(model.state_dict(), str(args.output / "model.pdparams"))
            (args.output / "config.json").write_text(json.dumps(config, indent=2))
        if args.device.startswith("gpu"):
            report["peak_allocated_bytes"] = paddle.device.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = paddle.device.cuda.max_memory_reserved()
        report["status"] = "passed"
        report["phase"] = "complete"
        save()
        print(
            f"Passed: initial loss {report['steps'][0]['loss']:.8g}, final {after:.8g}",
            flush=True,
        )
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "static", "stats", "data", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--no-recompute", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--lora-rank", type=int, default=0, help="0: full training; positive: LoRA"
    )
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--loss", choices=("mse", "mae", "crps"), default="mse")
    main(parser.parse_args())

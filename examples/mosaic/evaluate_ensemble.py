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

"""Compare base/LoRA ensemble scores on a real sample with identical noise.

Normalized weighted fair CRPS and ensemble-mean physical RMSE are different
metrics. A tiny fixed ensemble on one date is not a calibration/skill study.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import paddle
from convert_checkpoint import initialize_model
from evaluate_weather import weighted_mse
from lora import inject_lora
from lora import load_adapter
from losses import ensemble_forecast
from losses import weighted_loss
from train_smoke import variable_weights


def main(args):
    if args.members < 2:
        raise ValueError("Fair CRPS requires >=2 members")
    args.output.mkdir(parents=True, exist_ok=False)
    training = json.loads((args.adapter / "training.json").read_text())
    if training["status"] != "passed":
        raise ValueError("Require successfully validated training run")
    if (
        training["data"] == str(args.data.resolve())
        and training["sample_index"] == args.sample_index
    ):
        raise ValueError("Evaluation sample must not be the fixed training sample")
    paddle.set_device("cpu")
    cfg = json.loads((args.model / "config.json").read_text())
    model = initialize_model(cfg, args.static)
    state = paddle.load(str(args.model / "model.pdparams"))
    if set(state) != set(model.state_dict()):
        raise ValueError("Checkpoint key mismatch")
    model.set_state_dict(state)
    del state
    model.to(device=args.device)
    model.eval()
    paddle.set_device(args.device)
    with np.load(args.data) as data, np.load(args.stats) as stats, np.load(
        args.static
    ) as static:
        for key in ("longitude", "latitude"):
            np.testing.assert_allclose(data[key], static[key], atol=1e-5, rtol=0)
        i = args.sample_index
        if not 0 <= i < min(len(data["state"]), len(data["target"])):
            raise ValueError("sample-index outside data")
        mean, std = stats["state_mean"], stats["state_std"]
        raw, truth, latitude = (
            data["state"][i : i + 1],
            data["target"][i : i + 1, None],
            data["latitude"],
        )
        inputs = dict(
            state=paddle.to_tensor(((raw - mean) / std)[:, None].astype("float32")),
            time=paddle.to_tensor(data["time"][i : i + 1, None].astype("float32")),
            noise=paddle.to_tensor(
                np.random.default_rng(args.seed)
                .standard_normal((args.members, cfg["noise_dim"]))
                .astype("float32")
            ),
        )
        target = paddle.to_tensor(((truth - mean) / std).astype("float32"))
        lat = np.cos(np.deg2rad(latitude.astype("float64")))
        lat /= lat.mean()
        weights = paddle.to_tensor(
            (lat[:, None] * variable_weights(cfg["variables"])[None]).astype("float32")
        )[None, None, None]
    report = dict(
        data=str(args.data.resolve()),
        sample_index=i,
        members=args.members,
        seed=args.seed,
        model=str(args.model.resolve()),
        adapter=str(args.adapter.resolve()),
        scores={},
        scope="one-date same-noise base/adapter comparison; adjacent date may share context; not independent seasonal calibration or skill validation",
    )
    forecasts = {}
    for mode in ("base", "lora"):
        if mode == "lora":
            manifest = json.loads((args.adapter / "adapter_config.json").read_text())
            actual = inject_lora(model, manifest["rank"], manifest["alpha"])
            if actual != manifest:
                raise ValueError("Adapter target mismatch")
            load_adapter(model, args.adapter / "adapter.pdparams")
            model.eval()
        with paddle.no_grad():
            prediction = ensemble_forecast(model, inputs, args.members, False)
            normalized = prediction.numpy()
            if not np.isfinite(normalized).all():
                raise ValueError("Non-finite ensemble prediction")
            physical_mean = (normalized.astype("float64") * std + mean).mean(
                axis=1, keepdims=True
            )
            rmse = np.sqrt(weighted_mse(physical_mean, truth, latitude))
            report["scores"][mode] = dict(
                normalized_weighted_fair_crps=float(
                    weighted_loss(prediction, target, weights, "crps")
                ),
                normalized_weighted_member_mse=float(
                    weighted_loss(prediction, target, weights, "mse")
                ),
                physical_ensemble_mean_rmse=dict(
                    zip(cfg["variables"], map(float, rmse))
                ),
                normalized_mean_spread=float(normalized.std(axis=1).mean()),
            )
            forecasts[mode] = normalized
    base, lora = [report["scores"][mode] for mode in ("base", "lora")]
    regression = {
        name: 100
        * (
            lora["physical_ensemble_mean_rmse"][name]
            / base["physical_ensemble_mean_rmse"][name]
            - 1
        )
        for name in cfg["variables"]
    }
    report["adapter_vs_base"] = dict(
        worst_mean_rmse_regression_percent=max(regression.values()),
        variables=regression,
        normalized_fair_crps_change=lora["normalized_weighted_fair_crps"]
        - base["normalized_weighted_fair_crps"],
    )
    np.savez(args.output / "ensembles.npz", **forecasts)
    (args.output / "ensemble.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["adapter_vs_base"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "static", "stats", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--members", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    main(parser.parse_args())

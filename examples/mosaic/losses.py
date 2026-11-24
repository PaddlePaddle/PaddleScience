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

"""Weighted deterministic and fair ensemble scores, matching Mosaic losses.py."""

import paddle


def weighted_loss(prediction, target, weights, kind="mse"):
    """[B,E,lon,lat,C] forecasts and [B,1,lon,lat,C] normalized truth.

    Fair CRPS excludes self-pairs with denominator E*(E-1), as the reference.
    Accumulate unordered pairs instead of allocating an E by E spatial tensor.
    A finite ensemble's fair CRPS estimate need not be nonnegative.
    """
    if len(prediction.shape) != 5 or target.shape != [
        prediction.shape[0],
        1,
        *prediction.shape[2:],
    ]:
        raise ValueError("Require forecast [B,E,lon,lat,C] and target [B,1,lon,lat,C]")
    if kind not in ("mse", "mae", "crps"):
        raise ValueError(f"Unknown loss {kind}")
    error = prediction - target
    if kind == "mse":
        return (error.square() * weights).mean()
    if kind == "mae":
        return (error.abs() * weights).mean()
    members = prediction.shape[1]
    if members < 2:
        raise ValueError("Fair CRPS requires at least two ensemble members")
    score = error.abs().mean(axis=1, keepdim=True)
    for first in range(members):
        for second in range(first + 1, members):
            score = score - (
                prediction[:, first : first + 1] - prediction[:, second : second + 1]
            ).abs() / (members * (members - 1))
    return (score * weights).mean()


def ensemble_forecast(model, inputs, members, recompute_members=True):
    """Sequential explicit-noise members; optionally release each member graph.

    Each call uses the single-member model, avoiding ensemble-scaled attention
    tensors. Outer reentrant recomputation bounds retained activation memory.
    """
    if members < 1 or model.num_noise_samples != 1:
        raise ValueError("Require positive members and single-member base model")
    state, phase, noise = inputs["state"], inputs["time"], inputs["noise"]
    if state.shape[1] != 1:
        raise ValueError(
            "Ensemble training helper supports one forecast context per batch"
        )
    if noise.shape[0] != members * state.shape[0] * state.shape[1]:
        raise ValueError("Noise batch does not match requested ensemble")
    batch = state.shape[0] * state.shape[1]

    def predict(x, time, draw):
        return model({"state": x, "time": time, "noise": draw})["forecast"]

    outputs = []
    for member in range(members):
        draw = noise[member * batch : (member + 1) * batch]
        if recompute_members and paddle.is_grad_enabled():
            from paddle.distributed.fleet.utils import recompute

            state.stop_gradient = False
            outputs.append(recompute(predict, state, phase, draw, use_reentrant=True))
        else:
            outputs.append(predict(state, phase, draw))
    # Preserve model ordering [B,N*E,...], with ensembles inside each N slot.
    stacked = paddle.stack(outputs, axis=2)
    return stacked.flatten(start_axis=1, stop_axis=2)

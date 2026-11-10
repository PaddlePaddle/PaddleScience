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

"""Framework-independent direct-state ERA5 rollout; no target feedback."""

import numpy as np


def validate_era5_sequence(data, index, steps):
    if steps < 1 or index < 0 or index + steps > len(data["target"]):
        raise ValueError("Invalid rollout range")
    for offset in range(1, steps):
        # This extractor stores overlapping 24h windows. Check their consistency
        # without feeding any of these future observations to the predictor.
        np.testing.assert_array_equal(
            data["state"][index + offset, 0], data["state"][index + offset - 1, 1]
        )
        np.testing.assert_array_equal(
            data["state"][index + offset, 1], data["target"][index + offset - 1]
        )


def forecast_rollout(predict, history, phase, mean, std, noise_dim, seed, steps):
    if steps < 1:
        raise ValueError("rollout steps must be positive")
    history = history.astype("float32").copy()
    phase = phase.astype("float32").copy()
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(steps):
        normalized = (history - mean) / std
        noise = rng.standard_normal((len(history), noise_dim)).astype("float32")
        result = np.asarray(predict(normalized, phase, noise)).astype("float32")
        expected = (len(history), 1, *history.shape[-3:])
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError("Invalid rollout prediction shape or non-finite output")
        predictions.append(result)
        # Match the source direct-state physical-space feedback, not teacher forcing.
        physical = result * std + mean
        history = np.concatenate([history[:, 1:], physical], axis=1)
        phase += np.asarray([1.0, 4.0 / 365.25], dtype="float32")
    return np.concatenate(predictions, axis=1)

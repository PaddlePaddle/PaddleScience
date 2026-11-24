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

"""Independent checks for weather metric weighting and invalid inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/mosaic"))
import paddle  # noqa: E402
from evaluate_seasons import aggregate_reports  # noqa: E402
from evaluate_weather import weighted_mse  # noqa: E402
from losses import weighted_loss  # noqa: E402
from presets import VARIABLES  # noqa: E402
from rollout import forecast_rollout  # noqa: E402
from rollout import validate_era5_sequence  # noqa: E402


class TestMosaicWeather(unittest.TestCase):
    def test_fair_crps_value_and_gradients(self):
        paddle.set_device("cpu")
        paddle.seed(42)
        prediction = paddle.randn([2, 3, 4, 3, 2])
        prediction.stop_gradient = False
        target = paddle.randn([2, 1, 4, 3, 2])
        weights = paddle.rand([1, 1, 1, 3, 2])
        actual = weighted_loss(prediction, target, weights, "crps")
        pairwise = (prediction.unsqueeze(2) - prediction.unsqueeze(1)).abs()
        expected = (
            (prediction - target).abs().mean(axis=1, keepdim=True) * weights
        ).mean() - (pairwise.sum(axis=(1, 2)).unsqueeze(1) * weights).mean() / 12
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6)
        first = paddle.grad(actual, prediction, retain_graph=True)[0]
        second = paddle.grad(expected, prediction)[0]
        np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-7)
        for kind in ("mse", "mae"):
            error = prediction - target
            expected = (
                (error.square() if kind == "mse" else error.abs()) * weights
            ).mean()
            np.testing.assert_allclose(
                weighted_loss(prediction, target, weights, kind).numpy(),
                expected.numpy(),
            )
        with self.assertRaises(ValueError):
            weighted_loss(prediction[:, :1], target, weights, "crps")

    def test_seasonal_aggregation_and_case_guard(self):
        def report(actual, reference):
            return {
                "variables": {
                    name: dict(
                        actual_rmse=actual,
                        reference_rmse=reference,
                        persistence_rmse=10.0,
                    )
                    for name in VARIABLES
                }
            }

        result = aggregate_reports([report(1.0, 1.0), report(3.0, 2.0)], 40)
        self.assertAlmostEqual(
            result["variables"][VARIABLES[0]]["actual_rmse"], np.sqrt(5)
        )
        self.assertAlmostEqual(
            result["variables"][VARIABLES[0]]["reference_rmse"], np.sqrt(2.5)
        )
        # Aggregate regression is 41.4%, but the second individual case is 50%.
        self.assertAlmostEqual(result["worst_case_regression_percent"], 50)
        self.assertFalse(result["every_case_variable_within_tolerance"])
        result = aggregate_reports([report(1.0, 1.0), report(3.0, 2.0)], 45)
        self.assertTrue(result["aggregate_all_variables_within_tolerance"])
        self.assertFalse(result["every_case_variable_within_tolerance"])
        for reports, tolerance in (
            ([], 1),
            ([report(1, 0)], 1),
            ([report(float("nan"), 1)], 1),
            ([report(1, 1)], -1),
        ):
            with self.assertRaises(ValueError):
                aggregate_reports(reports, tolerance)

    def test_rollout_sequence_validation(self):
        data = dict(
            state=np.array([[0, 1], [1, 2], [2, 3]]), target=np.array([2, 3, 4])
        )
        validate_era5_sequence(data, 0, 3)
        with self.assertRaises(ValueError):
            validate_era5_sequence(data, 0, 4)
        data["state"][1, 1] = 100
        with self.assertRaises(AssertionError):
            validate_era5_sequence(data, 0, 3)

    def test_rollout_feedback_and_noise(self):
        calls = []

        def predict(history, phase, noise):
            calls.append((history.copy(), phase.copy(), noise.copy()))
            return history[:, -1:] + 1

        history = np.zeros((1, 2, 2, 3, 82), dtype="float32")
        phase = np.zeros((1, 2), dtype="float32")
        result = forecast_rollout(
            predict, history, phase, np.zeros(82), np.ones(82), 4, 2026, 3
        )
        np.testing.assert_allclose(result[:, :, 0, 0, 0], [[1, 2, 3]])
        np.testing.assert_array_equal(calls[1][0][:, -1], 1)
        np.testing.assert_allclose(calls[2][1], [[2, 8 / 365.25]])
        self.assertFalse(np.array_equal(calls[0][2], calls[1][2]))
        expected_noise = (
            np.random.default_rng(2026).standard_normal((1, 4)).astype("float32")
        )
        np.testing.assert_array_equal(calls[0][2], expected_noise)
        np.testing.assert_array_equal(history, 0)

    def test_cosine_weighted_mse(self):
        prediction = np.ones((2, 1, 4, 3, 82))
        target = np.zeros_like(prediction)
        np.testing.assert_allclose(weighted_mse(prediction, target, [-60, 0, 60]), 1)
        prediction.fill(0)
        prediction[..., 1, :] = 2
        # Equator carries half the total weight; squared error there is four.
        np.testing.assert_allclose(weighted_mse(prediction, target, [-60, 0, 60]), 2)

    def test_invalid_inputs(self):
        prediction = np.ones((1, 1, 4, 3, 82))
        with self.assertRaises(ValueError):
            weighted_mse(prediction, prediction[..., :1], [-60, 0, 60])
        with self.assertRaises(ValueError):
            weighted_mse(prediction, prediction, [0, 60])
        prediction[..., 0] = float("nan")
        with self.assertRaises(ValueError):
            weighted_mse(prediction, prediction, [-60, 0, 60])


if __name__ == "__main__":
    unittest.main()

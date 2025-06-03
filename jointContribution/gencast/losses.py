# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss functions (and terms for use in loss functions) used for weather."""

from typing import Dict
from typing import Tuple

import numpy as np
import paddle
import xarray
from typing_extensions import Protocol

LossAndDiagnostics = tuple[xarray.DataArray, xarray.Dataset]


class LossFunction(Protocol):
    """A loss function.

    This is a protocol so it's fine to use a plain function which 'quacks like'
    this. This is just to document the interface.
    """

    def __call__(
        self, predictions: xarray.Dataset, targets: xarray.Dataset, **optional_kwargs
    ) -> LossAndDiagnostics:
        """Computes a loss function.

        Args:
            predictions: Dataset of predictions.
            targets: Dataset of targets.
            **optional_kwargs: Implementations may support extra optional kwargs.

        Returns:
            loss: A DataArray with dimensions ('batch',) containing losses for each
                element of the batch. These will be averaged to give the final
                loss, locally and across replicas.
            diagnostics: Mapping of additional quantities to log by name alongside the
                loss. These will will typically correspond to terms in the loss. They
                should also have dimensions ('batch',) and will be averaged over the
                batch before logging.
        """


def xarray_to_tensor(data: xarray.DataArray) -> paddle.Tensor:
    """Convert xarray.DataArray to paddle.Tensor and move time into batch if needed."""
    arr = paddle.to_tensor(data.values, dtype="float32")
    # Collapse (batch, time) into single batch dimension
    if "time" in data.dims:
        batch_dim = data.sizes["batch"] * data.sizes["time"]
        new_shape = (batch_dim,) + tuple(arr.shape[2:])
        return arr.reshape(new_shape)
    return arr


def broadcast_weights(shape, lat_weights, level_weights=None):
    """Returns a broadcasted weight tensor matching given shape."""
    dims = len(shape)
    # Assume shape like (batch, level?, lat, lon)
    batch_shape = [1] * dims
    weight = lat_weights.reshape([*batch_shape[:-2], -1, 1])
    if level_weights is not None:
        weight = weight * level_weights.reshape([*batch_shape[:-3], -1, 1, 1])
    return weight


def normalized_latitude_weights(lat: np.ndarray) -> paddle.Tensor:
    lat_rad = paddle.to_tensor(np.deg2rad(lat), dtype="float32")
    d_lat = paddle.to_tensor(np.abs(lat[1] - lat[0]), dtype="float32")
    if paddle.any(
        paddle.isclose(paddle.abs(lat_rad), paddle.full_like(lat_rad, np.pi / 2))
    ):
        weights = paddle.cos(lat_rad) * paddle.sin(d_lat / 2)
        weights[0] = weights[-1] = paddle.sin(d_lat / 4) ** 2
    else:
        weights = paddle.cos(lat_rad)
    return weights / weights.mean()


def normalized_level_weights(level: np.ndarray) -> paddle.Tensor:
    level_tensor = paddle.to_tensor(level, dtype="float32")
    return level_tensor / level_tensor.mean()


def weighted_mse_loss_from_xarray(
    predictions: paddle.Tensor,
    targets: xarray.Dataset,
    per_variable_weights: Dict[str, float],
) -> Tuple[paddle.Tensor, Dict[str, paddle.Tensor]]:
    """
    Compute latitude- and pressure-weighted MSE loss from xarray datasets.
    """
    lat = targets["lat"].values
    level = targets["level"].values if "level" in targets.coords else None

    lat_weights = normalized_latitude_weights(lat)
    level_weights = normalized_level_weights(level) if level is not None else None

    per_variable_losses = {}
    total_loss = None

    channels = {
        "10m_u_component_of_wind": 1,
        "10m_v_component_of_wind": 1,
        "2m_temperature": 1,
        "geopotential": 13,
        "mean_sea_level_pressure": 1,
        "sea_surface_temperature": 1,
        "specific_humidity": 13,
        "temperature": 13,
        "total_precipitation_12hr": 1,
        "u_component_of_wind": 13,
        "v_component_of_wind": 13,
        "vertical_velocity": 13,
    }
    start_index = 0
    for var_name, size in channels.items():
        end_index = start_index + size
        if var_name not in targets.data_vars:
            continue

        target_tensor = xarray_to_tensor(targets[var_name])
        if len(target_tensor.shape) == 3:
            pred_tensor = predictions[:, :, :, start_index:end_index].transpose(
                [2, 3, 0, 1]
            )[0]
        else:
            pred_tensor = predictions[:, :, :, start_index:end_index].transpose(
                [2, 3, 0, 1]
            )

        assert pred_tensor.shape == target_tensor.shape

        var_loss = (pred_tensor - target_tensor) ** 2

        # Determine if it's a 3D (with level) or 2D variable
        dims = targets[var_name].dims
        has_level = "level" in dims

        weight_tensor = broadcast_weights(
            var_loss.shape, lat_weights, level_weights if has_level else None
        )

        weighted_loss = var_loss * weight_tensor
        loss_reduced = weighted_loss.mean(axis=tuple(range(1, weighted_loss.ndim)))
        per_variable_losses[var_name] = loss_reduced  # shape: (batch,)

        weight = per_variable_weights.get(var_name, 1.0)
        if total_loss is None:
            total_loss = weight * loss_reduced
        else:
            total_loss = total_loss + weight * loss_reduced

        start_index = end_index

    final_loss = total_loss.mean()
    return final_loss, per_variable_losses

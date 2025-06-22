# Copyright 2024 DeepMind Technologies Limited.
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
"""Denoising diffusion models based on the framework of [1].

Throughout we will refer to notation and equations from [1].

  [1] Elucidating the Design Space of Diffusion-Based Generative Models
  Karras, Aittala, Aila and Laine, 2022
  https://arxiv.org/abs/2206.00364
"""

from typing import Optional

import denoiser
import dpm_solver_plus_plus_2s
import losses
import numpy as np
import paddle
import paddle.nn as nn
import samplers_utils
import xarray as xr
from graphcast import datasets


class GenCast(nn.Layer):
    """Predictor for a denoising diffusion model following the framework of [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    Karras, Aittala, Aila and Laine, 2022
    https://arxiv.org/abs/2206.00364

    Unlike the paper, we have a conditional model and our denoising function
    conditions on previous timesteps.

    As the paper demonstrates, the sampling algorithm can be varied independently
    of the denoising model and its training procedure, and it is separately
    configurable here.
    """

    def __init__(
        self,
        cfg,
    ):
        """Constructs GenCast."""
        super(GenCast, self).__init__()

        self._denoiser = denoiser.Denoiser(cfg)
        self._sampler_config = cfg.sampler_config
        self._sampler = None
        self._noise_config = cfg.noise_config
        self.cfg = cfg

    def _c_in(self, noise_scale: xr.DataArray) -> xr.DataArray:
        """Scaling applied to the noisy targets input to the underlying network."""
        return (noise_scale**2 + 1) ** -0.5

    def _c_out(self, noise_scale: xr.DataArray) -> xr.DataArray:
        """Scaling applied to the underlying network's raw outputs."""
        return noise_scale * (noise_scale**2 + 1) ** -0.5

    def _c_skip(self, noise_scale: xr.DataArray) -> xr.DataArray:
        """Scaling applied to the skip connection."""
        return 1 / (noise_scale**2 + 1)

    def _loss_weighting(self, noise_scale: xr.DataArray) -> xr.DataArray:
        r"""The loss weighting \lambda(\sigma) from the paper."""
        return self._c_out(noise_scale) ** -2

    def _preconditioned_denoiser(
        self,
        inputs: xr.Dataset,
        noisy_targets: xr.Dataset,
        noise_levels: xr.DataArray,
        forcings: Optional[xr.Dataset] = None,
        **kwargs
    ) -> xr.Dataset:
        """The preconditioned denoising function D from the paper (Eqn 7)."""
        # Convert xarray DataArray to Paddle tensor for operations
        raw_predictions, grid_node_outputs = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets * self._c_in(noise_levels),
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs
        )

        stacked_noisy_targets = datasets.dataset_to_stacked(noisy_targets)
        stacked_noisy_targets = stacked_noisy_targets.transpose("lat", "lon", ...)

        out = grid_node_outputs * paddle.to_tensor(self._c_out(noise_levels).data)
        skip = paddle.to_tensor(
            stacked_noisy_targets.data * self._c_skip(noise_levels).data
        )
        grid_node_outputs = out + skip

        return (
            raw_predictions * self._c_out(noise_levels)
            + noisy_targets * self._c_skip(noise_levels),
            grid_node_outputs,
        )

    def loss(
        self,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        forcings: Optional[xr.Dataset] = None,
    ):

        if self._noise_config is None:
            raise ValueError("Noise config must be specified to train GenCast.")

        grid_node_outputs, denoised_predictions, noise_levels = self.forward(
            inputs, targets, forcings
        )

        loss, diagnostics = losses.weighted_mse_loss_from_xarray(
            grid_node_outputs,
            targets,
            # Weights are same as we used for GraphCast.
            per_variable_weights={
                # Any variables not specified here are weighted as 1.0.
                # A single-level variable, but an important headline variable
                # and also one which we have struggled to get good performance
                # on at short lead times, so leaving it weighted at 1.0, equal
                # to the multi-level variables:
                "2m_temperature": 1.0,
                # New single-level variables, which we don't weight too highly
                # to avoid hurting performance on other variables.
                "10m_u_component_of_wind": 0.1,
                "10m_v_component_of_wind": 0.1,
                "mean_sea_level_pressure": 0.1,
                "sea_surface_temperature": 0.1,
                "total_precipitation_12hr": 0.1,
            },
        )
        loss *= paddle.to_tensor(self._loss_weighting(noise_levels).data)
        return loss, diagnostics

    def forward(self, inputs, targets_template, forcings=None, **kwargs):
        if self.cfg.mode == "eval":
            if self._sampler is None:
                self._sampler = dpm_solver_plus_plus_2s.Sampler(
                    self._preconditioned_denoiser, **self._sampler_config
                )
            return self._sampler(inputs, targets_template, forcings, **kwargs)
        if self.cfg.mode == "train":
            # Sample noise levels:
            batch_size = inputs.sizes["batch"]
            noise_levels = xr.DataArray(
                data=samplers_utils.rho_inverse_cdf(
                    min_value=self._noise_config.training_min_noise_level,
                    max_value=self._noise_config.training_max_noise_level,
                    rho=self._noise_config.training_noise_level_rho,
                    cdf=np.random.uniform(size=(batch_size,)).astype("float32"),
                ),
                dims=("batch",),
            )

            # Sample noise and apply it to targets:
            noise = (
                samplers_utils.spherical_white_noise_like(targets_template)
                * noise_levels
            )

            noisy_targets = targets_template + noise

            denoised_predictions, grid_node_outputs = self._preconditioned_denoiser(
                inputs, noisy_targets, noise_levels, forcings
            )
            return grid_node_outputs, denoised_predictions, noise_levels

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
import paddle.nn as nn
import xarray as xr


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
        raw_predictions = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets * self._c_in(noise_levels),
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs
        )

        return raw_predictions * self._c_out(
            noise_levels
        ) + noisy_targets * self._c_skip(noise_levels)

    def forward(self, inputs, targets_template, forcings=None, **kwargs):

        if self._sampler is None:
            self._sampler = dpm_solver_plus_plus_2s.Sampler(
                self._preconditioned_denoiser, **self._sampler_config
            )
        return self._sampler(inputs, targets_template, forcings, **kwargs)

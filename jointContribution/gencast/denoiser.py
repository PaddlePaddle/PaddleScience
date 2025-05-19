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
"""Support for wrapping a general Predictor to act as a Denoiser."""

import copy
import math
import os
import pickle
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import paddle.nn as nn
import xarray as xr
from graphcast import datasets
from graphcast import graphcast
from graphcast import graphtype
from graphcast import utils


class FourierFeaturesMLP(nn.Layer):
    """A simple MLP applied to Fourier features of values or their logarithms."""

    def __init__(
        self,
        base_period: float,
        num_frequencies: int,
        output_sizes: Sequence[int],
        apply_log_first: bool = False,
        w_init: Optional[nn.initializer.Initializer] = None,
        activation: Optional[nn.Layer] = nn.GELU(),
        **mlp_kwargs,
    ):
        """Initializes the module.

        Args:
        base_period:
            See model_utils.fourier_features. Note this would apply to log inputs if
            apply_log_first is used.
        num_frequencies:
            See model_utils.fourier_features.
        output_sizes:
            Layer sizes for the MLP.
        apply_log_first:
            Whether to take the log of the inputs before computing Fourier features.
        w_init:
            Weights initializer for the MLP, default setting aims to produce
            approx unit-variance outputs given the input sin/cos features.
        activation:
        **mlp_kwargs:
            Further settings for the MLP.
        """
        super(FourierFeaturesMLP, self).__init__()
        self._base_period = base_period
        self._num_frequencies = num_frequencies
        self._apply_log_first = apply_log_first

        # Creating MLP
        layers = []
        input_size = 2 * num_frequencies
        num_layers = len(output_sizes)
        for i, output_size in enumerate(output_sizes):
            limit = math.sqrt(6 / input_size)
            weight_attr = paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(low=-limit, high=limit)
            )
            linear_layer = nn.Linear(input_size, output_size, weight_attr=weight_attr)
            layers.append(linear_layer)
            if i < num_layers - 1:
                layers.append(activation)
            input_size = output_size

        self._mlp = nn.Sequential(*layers)

    def forward(self, values: paddle.Tensor) -> paddle.Tensor:
        if self._apply_log_first:
            values = paddle.log(values)
        features = utils.fourier_features(
            values, self._base_period, self._num_frequencies
        )

        return self._mlp(features)


class Denoiser(nn.Layer):
    """Wraps a general deterministic Predictor to act as a Denoiser.

    This passes an encoding of the noise level as an additional input to the
    Predictor as an additional input 'noise_level_encodings' with shape
    ('batch', 'noise_level_encoding_channels'). It passes the noisy_targets as
    additional forcings (since they are also per-target-timestep data that the
    predictor needs to condition on) with the same names as the original target
    variables.
    """

    def __init__(
        self,
        cfg,
    ):
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self._predictor = graphcast.GraphCastNet(
            config=cfg.denoiser_architecture_config,
        )

        self._noise_level_encoder = FourierFeaturesMLP(**cfg.noise_encoder_config)

    def forward(
        self,
        inputs: xr.Dataset,
        noisy_targets: xr.Dataset,
        noise_levels: xr.DataArray,
        forcings: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> xr.Dataset:

        if forcings is None:
            forcings = xr.Dataset()
        forcings = forcings.assign(**noisy_targets)

        if noise_levels.dims != ("batch",):
            raise ValueError("noise_levels expected to be shape (batch,).")

        noise_level_encodings = self._noise_level_encoder(
            paddle.to_tensor(noise_levels.values)
        )

        stacked_inputs = datasets.dataset_to_stacked(inputs)

        stacked_forcings = datasets.dataset_to_stacked(forcings)
        stacked_inputs = xr.concat([stacked_inputs, stacked_forcings], dim="channels")

        stacked_inputs = stacked_inputs.transpose("lat", "lon", ...)
        lat_dim, lon_dim, batch_dim, feat_dim = stacked_inputs.shape
        stacked_inputs = stacked_inputs.data.reshape(lat_dim * lon_dim, batch_dim, -1)

        graph_template_path = os.path.join(
            "data", "template_graph", f"{self.cfg.type}.pkl"
        )
        if os.path.exists(graph_template_path):
            graph_template = pickle.load(open(graph_template_path, "rb"))
        else:
            graph_template = graphtype.GraphGridMesh(
                self.cfg.denoiser_architecture_config
            )
        graph = copy.deepcopy(graph_template)

        graph.grid_node_feat = np.concatenate(
            [stacked_inputs, graph.grid_node_feat], axis=-1
        )
        mesh_node_feat = np.zeros([graph.mesh_num_nodes, batch_dim, feat_dim])
        graph.mesh_node_feat = np.concatenate(
            [mesh_node_feat, graph.mesh_node_feat], axis=-1
        )
        graph.global_norm_conditioning = noise_level_encodings

        predictor = self._predictor(graph=graphtype.convert_np_to_tensor(graph))

        grid_node_outputs = predictor.grid_node_feat
        raw_predictions = predictor.grid_node_outputs_to_prediction(
            grid_node_outputs, noisy_targets
        )

        resolution = self.cfg.denoiser_architecture_config.resolution
        grid_lat = np.arange(-90.0, 90.0 + resolution, resolution).astype(np.float32)
        grid_lon = np.arange(0.0, 360.0, resolution).astype(np.float32)
        grid_shape = [grid_lat.shape[0], grid_lon.shape[0]]
        grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
            grid_shape + grid_node_outputs.shape[1:]
        )

        return raw_predictions, grid_outputs_lat_lon_leading

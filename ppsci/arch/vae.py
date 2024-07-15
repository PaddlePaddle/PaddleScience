# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Tuple

import paddle
import paddle.nn as nn

from ppsci.arch import base


class AutoEncoder(base.Arch):
    """
    AutoEncoder is a class that represents an autoencoder neural network model.

    Args:
        input_keys (Tuple[str, ...]): A tuple of input keys.
        output_keys (Tuple[str, ...]): A tuple of output keys.
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space.
        hidden_dim (int): The dimension of the hidden layer.

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.AutoEncoder(
        ...    input_keys=("input1",),
        ...    output_keys=("mu", "log_sigma", "decoder_z",),
        ...    input_dim=100,
        ...    latent_dim=50,
        ...    hidden_dim=200
        ... )
        >>> input_dict = {"input1": paddle.rand([200, 100]),}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["mu"].shape)
        [200, 50]
        >>> print(output_dict["log_sigma"].shape)
        [200, 50]
        >>> print(output_dict["decoder_z"].shape)
        [200, 100]
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ):
        super(AutoEncoder, self).__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        # encoder
        self._encoder_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self._encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self._encoder_log_sigma = nn.Linear(hidden_dim, latent_dim)

        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encoder(self, x):
        h = self._encoder_linear(x)
        mu = self._encoder_mu(h)
        log_sigma = self._encoder_log_sigma(h)
        return mu, log_sigma

    def decoder(self, x):
        return self._decoder(x)

    def forward_tensor(self, x):
        mu, log_sigma = self.encoder(x)
        z = mu + paddle.randn(mu.shape) * paddle.exp(log_sigma)
        return mu, log_sigma, self.decoder(z)

    def forward(self, x):
        x = self.concat_to_tensor(x, self.input_keys, axis=-1)
        mu, log_sigma, decoder_z = self.forward_tensor(x)
        result_dict = {
            self.output_keys[0]: mu,
            self.output_keys[1]: log_sigma,
            self.output_keys[2]: decoder_z,
        }
        return result_dict

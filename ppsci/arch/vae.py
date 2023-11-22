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


# copy from AISTUDIO
class AutoEncoder(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        input_dim,
        latent_dim,
        hidden_dim,
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

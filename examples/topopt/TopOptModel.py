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

import paddle
from paddle import nn

import ppsci


# NCHW data format
class TopOptNN(ppsci.arch.UNetEx):
    def __init__(
        self,
        input_key="input",
        output_key="output",
        in_channel=2,
        out_channel=1,
        kernel_size=3,
        filters=(16, 32, 64),
        layers=2,
        channel_sampler=lambda: 1,
        weight_norm=False,
        batch_norm=False,
        activation=nn.ReLU,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            filters=filters,
            layers=layers,
            weight_norm=weight_norm,
            batch_norm=batch_norm,
            activation=activation,
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filters = filters
        self.channel_sampler = channel_sampler

        # Modify Layers
        self.encoder[1] = nn.Sequential(
            nn.MaxPool2D(self.in_channel, padding="SAME"),
            self.encoder[1][0],
            nn.Dropout2D(0.1),
            self.encoder[1][1],
        )
        self.encoder[2] = nn.Sequential(
            nn.MaxPool2D(2, padding="SAME"), self.encoder[2]
        )
        # Conv2D used in reference code in decoder
        self.decoders[0] = nn.Sequential(
            nn.Conv2D(
                self.filters[-1], self.filters[-1], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
            nn.Conv2D(
                self.filters[-1], self.filters[-1], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
        )
        self.decoders[1] = nn.Sequential(
            nn.Conv2D(
                sum(self.filters[-2:]), self.filters[-2], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
            nn.Dropout2D(0.1),
            nn.Conv2D(
                self.filters[-2], self.filters[-2], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
        )
        self.decoders[2] = nn.Sequential(
            nn.Conv2D(
                sum(self.filters[:-1]), self.filters[-3], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
            nn.Conv2D(
                self.filters[-3], self.filters[-3], kernel_size=3, padding="SAME"
            ),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2D(
                self.filters[-3], self.out_channel, kernel_size=3, padding="SAME"
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        k = self.channel_sampler()
        x1 = x[self.input_keys[0]][:, k, :, :]
        x2 = x[self.input_keys[0]][:, k - 1, :, :]
        x = paddle.stack((x1, x1 - x2), axis=1)

        # encode
        upsampling_size = []
        skip_connection = []
        n_encoder = len(self.encoder)
        for i in range(n_encoder):
            x = self.encoder[i](x)
            if i is not (n_encoder - 1):
                upsampling_size.append(x.shape[-2:])
                skip_connection.append(x)

        # decode
        n_decoder = len(self.decoders)
        for i in range(n_decoder):
            x = self.decoders[i](x)
            if i is not (n_decoder - 1):
                up_size = upsampling_size.pop()
                x = nn.UpsamplingNearest2D(up_size)(x)
                skip_output = skip_connection.pop()
                x = paddle.concat((skip_output, x), axis=1)

        out = self.output(x)
        return {self.output_keys[0]: out}

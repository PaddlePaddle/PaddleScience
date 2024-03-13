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
    """Neural network for Topology Optimization, inherited from `ppsci.arch.UNetEx`

    [Sosnovik, I., & Oseledets, I. (2019). Neural networks for topology optimization. Russian Journal of Numerical Analysis and Mathematical Modelling, 34(4), 215-223.](https://arxiv.org/pdf/1709.09578)

    Args:
        input_key (str): Name of function data for input.
        output_key (str): Name of function data for output.
        in_channel (int): Number of channels of input.
        out_channel (int): Number of channels of output.
        kernel_size (int, optional): Size of kernel of convolution layer. Defaults to 3.
        filters (Tuple[int, ...], optional): Number of filters. Defaults to (16, 32, 64).
        layers (int, optional): Number of encoders or decoders. Defaults to 3.
        channel_sampler (callable, optional): The sampling function for the initial iteration time
                (corresponding to the channel number of the input) of SIMP algorithm. The default value
                is None, when it is None, input for the forward method should be sampled and prepared
                with the shape of [batch, 2, height, width] before passing to forward method.
        weight_norm (bool, optional): Whether use weight normalization layer. Defaults to True.
        batch_norm (bool, optional): Whether add batch normalization layer. Defaults to True.
        activation (Type[nn.Layer], optional): Name of activation function. Defaults to nn.ReLU.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.ppsci.arch.TopOptNN("input", "output", 2, 1, 3, (16, 32, 64), 2, lambda: 1, Flase, False)
    """

    def __init__(
        self,
        input_key="input",
        output_key="output",
        in_channel=2,
        out_channel=1,
        kernel_size=3,
        filters=(16, 32, 64),
        layers=2,
        channel_sampler=None,
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
        self.activation = activation

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
            self.activation(),
            nn.Conv2D(
                self.filters[-1], self.filters[-1], kernel_size=3, padding="SAME"
            ),
            self.activation(),
        )
        self.decoders[1] = nn.Sequential(
            nn.Conv2D(
                sum(self.filters[-2:]), self.filters[-2], kernel_size=3, padding="SAME"
            ),
            self.activation(),
            nn.Dropout2D(0.1),
            nn.Conv2D(
                self.filters[-2], self.filters[-2], kernel_size=3, padding="SAME"
            ),
            self.activation(),
        )
        self.decoders[2] = nn.Sequential(
            nn.Conv2D(
                sum(self.filters[:-1]), self.filters[-3], kernel_size=3, padding="SAME"
            ),
            self.activation(),
            nn.Conv2D(
                self.filters[-3], self.filters[-3], kernel_size=3, padding="SAME"
            ),
            self.activation(),
        )
        self.output = nn.Sequential(
            nn.Conv2D(
                self.filters[-3], self.out_channel, kernel_size=3, padding="SAME"
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.channel_sampler is not None:
            SIMP_initial_iter_time = self.channel_sampler()  # channel k
            input_channel_k = x[self.input_keys[0]][:, SIMP_initial_iter_time, :, :]
            input_channel_k_minus_1 = x[self.input_keys[0]][
                :, SIMP_initial_iter_time - 1, :, :
            ]
            x = paddle.stack(
                (input_channel_k, input_channel_k - input_channel_k_minus_1), axis=1
            )
        else:
            x = x[self.input_keys[0]]
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

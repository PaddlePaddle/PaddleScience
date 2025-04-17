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

# Code is heavily based on paper "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", we use paddle to reproduce the results of the paper

import paddle


class MLP(paddle.nn.Layer):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = paddle.nn.LayerList()
        for j in range(self.n_layers):
            self.layers.append(
                paddle.nn.Linear(in_features=layers[j], out_features=layers[j + 1])
            )
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(
                        paddle.nn.BatchNorm1D(num_features=layers[j + 1])
                    )
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class PositionalEmbedding(paddle.nn.Layer):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = paddle.arange(start=0, end=self.num_channels // 2, dtype="float32")
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        paddle.cast(freqs, x.dtype)
        x = x.outer(y=freqs)
        x = paddle.concat(x=[x.cos(), x.sin()], axis=1)
        return x


class AdaIN(paddle.nn.Layer):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-05):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps
        if mlp is None:
            mlp = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=embed_dim, out_features=512),
                paddle.nn.GELU(),
                paddle.nn.Linear(in_features=512, out_features=2 * in_channels),
            )
        self.mlp = mlp
        self.embedding = None

    def update_embeddding(self, x):
        self.embedding = x.reshape([self.embed_dim])

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"
        num_or_sections = self.mlp(self.embedding).shape[0] // self.in_channels
        embedding_weight, embedding_bias = paddle.split(
            x=self.mlp(self.embedding), num_or_sections=num_or_sections, axis=0
        )
        weight_attr = paddle.ParamAttr(  # name="embedding_weight",
            initializer=paddle.nn.initializer.Assign(embedding_weight),
            # trainable=True,
        )
        bias_attr = paddle.ParamAttr(  # name="embedding_bias",
            initializer=paddle.nn.initializer.Assign(embedding_bias),
            # trainable=True
        )
        group_norm = paddle.nn.GroupNorm(
            num_groups=self.in_channels,
            num_channels=self.in_channels,
            epsilon=self.eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

        # print(group_norm(x).shape)
        return group_norm(x)

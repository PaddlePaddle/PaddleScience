# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Chengxi Zang

import numpy as np
import paddle
import paddle.nn as nn
from scipy import linalg as la


# logabs = lambda x: paddle.log(x=paddle.abs(x=x))
def logabs(x):
    return paddle.log(paddle.abs(x))


class ActNorm(nn.Layer):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = self.create_parameter(
            [1, in_channel, 1, 1],
            default_initializer=nn.initializer.Constant(value=0.0),
        )

        self.scale = self.create_parameter(
            [1, in_channel, 1, 1],
            default_initializer=nn.initializer.Constant(value=1.0),
        )

        self.register_buffer(
            name="initialized", tensor=paddle.to_tensor(data=0, dtype="uint8")
        )
        self.logdet = logdet

    def initialize(self, input):
        with paddle.no_grad():
            flatten = input.transpose(perm=[1, 0, 2, 3]).reshape(
                [tuple(input.shape)[1], -1]
            )
            mean = (
                flatten.mean(axis=1)
                .unsqueeze(axis=1)
                .unsqueeze(axis=2)
                .unsqueeze(axis=3)
                .transpose(perm=[1, 0, 2, 3])
            )
            std = (
                flatten.std(axis=1)
                .unsqueeze(axis=1)
                .unsqueeze(axis=2)
                .unsqueeze(axis=3)
                .transpose(perm=[1, 0, 2, 3])
            )
            paddle.assign(-mean, output=self.loc.data)
            paddle.assign(1 / (std + 1e-06), output=self.scale.data)

    def forward(self, input):
        _, _, height, width = tuple(input.shape)
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(value=1)
        log_abs = logabs(self.scale)
        logdet = height * width * paddle.sum(x=log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNorm2D(nn.Layer):
    def __init__(self, in_dim, logdet=True):
        super().__init__()
        self.loc = self.create_parameter(
            [1, in_dim, 1],
            default_initializer=nn.initializer.Constant(value=0.0),
        )

        self.scale = self.create_parameter(
            [1, in_dim, 1],
            default_initializer=nn.initializer.Constant(value=1.0),
        )

        self.register_buffer(
            name="initialized", tensor=paddle.to_tensor(data=0, dtype="uint8")
        )
        self.logdet = logdet

    def initialize(self, input):
        with paddle.no_grad():
            flatten = input.transpose(perm=[1, 0, 2]).reshape(
                [tuple(input.shape)[1], -1]
            )
            mean = (
                flatten.mean(axis=1)
                .unsqueeze(axis=1)
                .unsqueeze(axis=2)
                .transpose(perm=[1, 0, 2])
            )
            std = (
                flatten.std(axis=1)
                .unsqueeze(axis=1)
                .unsqueeze(axis=2)
                .transpose(perm=[1, 0, 2])
            )
            paddle.assign(-mean, output=self.loc.data)
            paddle.assign(1 / (std + 1e-06), output=self.scale.data)

    def forward(self, input):
        _, _, height = tuple(input.shape)
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(value=1)
        log_abs = logabs(self.scale)
        logdet = height * paddle.sum(x=log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Layer):
    def __init__(self, in_channel):
        super().__init__()
        weight = paddle.randn([in_channel, in_channel])
        q, _ = paddle.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = paddle.create_parameter(
            weight.shape,
            weight.numpy().dtype,
            default_initializer=nn.initializer.Assign(weight),
        )

    def forward(self, input):
        _, _, height, width = tuple(input.shape)
        out = nn.functional.conv2d(x=input, weight=self.weight)
        res = paddle.linalg.slogdet(self.weight.squeeze().astype(dtype="float64"))
        logdet = height * width * (res[0], res[1])[1].astype(dtype="float32")
        return out, logdet

    def reverse(self, output):
        return nn.functional.conv2d(
            x=output,
            weight=self.weight.squeeze().inverse().unsqueeze(axis=2).unsqueeze(axis=3),
        )


class InvConv2dLU(nn.Layer):
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = paddle.to_tensor(data=w_p)
        w_l = paddle.to_tensor(data=w_l)
        w_s = paddle.to_tensor(data=w_s)
        w_u = paddle.to_tensor(data=w_u)
        self.register_buffer(name="w_p", tensor=w_p)
        self.register_buffer(name="u_mask", tensor=paddle.to_tensor(data=u_mask))
        self.register_buffer(name="l_mask", tensor=paddle.to_tensor(data=l_mask))
        self.register_buffer(name="s_sign", tensor=paddle.sign(x=w_s))
        self.register_buffer(
            name="l_eye", tensor=paddle.eye(num_rows=tuple(l_mask.shape)[0])
        )
        self.w_l = paddle.create_parameter(
            w_l.shape,
            w_l.numpy().dtype,
            default_initializer=nn.initializer.Assign(w_l),
        )

        self.w_s = paddle.create_parameter(
            logabs(w_s).shape,
            logabs(w_s).numpy().dtype,
            default_initializer=nn.initializer.Assign(logabs(w_s)),
        )

        self.w_u = paddle.create_parameter(
            w_u.shape,
            w_u.numpy().dtype,
            default_initializer=nn.initializer.Assign(w_u),
        )

    def forward(self, input):
        _, _, height, width = tuple(input.shape)
        weight = self.calc_weight()
        out = nn.functional.conv2d(x=input, weight=weight)
        logdet = height * width * paddle.sum(x=self.w_s)
        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ (
                self.w_u * self.u_mask
                + paddle.diag(x=self.s_sign * paddle.exp(x=self.w_s))
            )
        )
        return weight.unsqueeze(axis=2).unsqueeze(axis=3)

    def reverse(self, output):
        weight = self.calc_weight()
        return nn.functional.conv2d(
            x=output,
            weight=weight.squeeze().inverse().unsqueeze(axis=2).unsqueeze(axis=3),
        )


class GraphLinear(nn.Layer):
    """Graph Linear layer.
    This function assumes its input is 3-dimensional. Or 4-dim or whatever, only last dim are changed
    Differently from :class:`nn.Linear`, it applies an affine
    transformation to the third axis of input `x`.
    Warning: original Chainer.link.Link use i.i.d. Gaussian initialization as default,
    while default nn.Linear initialization using init.kaiming_uniform_
    """

    def __init__(self, in_size, out_size, bias=True):
        super(GraphLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(
            in_features=in_size, out_features=out_size, bias_attr=bias
        )

    def forward(self, x):
        """Forward propagation.
        Args:
            x (:class:`chainer.Variable`, or :class:`numpy.ndarray` ):
                Input array that should be a float array whose ``ndim`` is 3.

                It represents a minibatch of atoms, each of which consists
                of a sequence of molecules. Each molecule is represented
                by integer IDs. The first axis is an index of atoms
                (i.e. minibatch dimension) and the second one an index
                of molecules.

        Returns:
            class:`chainer.Variable`:
                A 3-dimeisional array.

        """
        h = x
        h = h.reshape([-1, tuple(x.shape)[-1]])
        h = self.linear(h)
        h = h.reshape(tuple(tuple(x.shape)[:-1] + (self.out_size,)))
        return h


class GraphConv(nn.Layer):
    """
    graph convolution over batch and multi-graphs
    Args:
        in_channels:   e.g. 8
        out_channels:  e.g. 64
        num_edge_type (types of edges/bonds):  e.g. 4
    return:
        class:`chainer.Variable`:
    """

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        super(GraphConv, self).__init__()
        self.graph_linear_self = GraphLinear(in_channels, out_channels)
        self.graph_linear_edge = GraphLinear(in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        mb, node, ch = tuple(h.shape)
        hs = self.graph_linear_self(h)
        m = self.graph_linear_edge(h)
        m = m.reshape([mb, node, self.out_ch, self.num_edge_type])
        m = m.transpose(perm=[0, 3, 1, 2])
        hr = paddle.matmul(x=adj, y=m)
        hr = hr.sum(axis=1)
        return hs + hr

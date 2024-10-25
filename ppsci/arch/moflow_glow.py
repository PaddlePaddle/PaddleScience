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

import warnings

import paddle
import paddle.nn as nn

from ppsci.arch.moflow_basic import ActNorm
from ppsci.arch.moflow_basic import ActNorm2D
from ppsci.arch.moflow_basic import GraphConv
from ppsci.arch.moflow_basic import GraphLinear
from ppsci.arch.moflow_basic import InvConv2d
from ppsci.arch.moflow_basic import InvConv2dLU

warnings.filterwarnings(
    "ignore", message="when training, we now always track global mean and variance."
)


class AffineCoupling(nn.Layer):
    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=False):
        super(AffineCoupling, self).__init__()
        self.affine = affine
        self.layers = nn.LayerList()
        self.norms = nn.LayerList()
        self.mask_swap = mask_swap
        last_h = in_channel // 2
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
            vh = tuple(hidden_channels) + (in_channel // 2,)
        for h in vh:
            self.layers.append(
                nn.Conv2D(in_channels=last_h, out_channels=h, kernel_size=3, padding=1)
            )
            self.norms.append(nn.BatchNorm2D(num_features=h))
            last_h = h

    def forward(self, input):
        in_a, in_b = input.chunk(chunks=2, axis=1)
        if self.mask_swap:
            in_a, in_b = in_b, in_a
        if self.affine:
            s, t = self._s_t_function(in_a)
            out_b = (in_b + t) * s
            logdet = paddle.sum(
                x=paddle.log(x=paddle.abs(x=s)).reshape([tuple(input.shape)[0], -1]),
                axis=1,
            )
        else:
            _, t = self._s_t_function(in_a)
            out_b = in_b + t
            logdet = None
        if self.mask_swap:
            result = paddle.concat(x=[out_b, in_a], axis=1)
        else:
            result = paddle.concat(x=[in_a, out_b], axis=1)
        return result, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(chunks=2, axis=1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a
        if self.affine:
            s, t = self._s_t_function(out_a)
            in_b = out_b / s - t
        else:
            _, t = self._s_t_function(out_a)
            in_b = out_b - t
        if self.mask_swap:
            result = paddle.concat(x=[in_b, out_a], axis=1)
        else:
            result = paddle.concat(x=[out_a, in_b], axis=1)
        return result

    def _s_t_function(self, x):
        h = x
        for i in range(len(self.layers) - 1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            h = nn.functional.relu(x=h)
        h = self.layers[-1](h)
        s = None
        if self.affine:
            log_s, t = h.chunk(chunks=2, axis=1)
            s = nn.functional.sigmoid(x=log_s)
        else:
            t = h
        return s, t


class GraphAffineCoupling(nn.Layer):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine
        self.hidden_dim_gnn = hidden_dim_dict["gnn"]
        self.hidden_dim_linear = hidden_dim_dict["linear"]
        self.net = nn.LayerList()
        self.norm = nn.LayerList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1D(num_features=n_node))
            last_dim = out_dim
        self.net_lin = nn.LayerList()
        self.norm_lin = nn.LayerList()
        for out_dim in self.hidden_dim_linear:
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(nn.BatchNorm1D(num_features=n_node))
            last_dim = out_dim
        if affine:
            self.net_lin.append(GraphLinear(last_dim, in_dim * 2))
        else:
            self.net_lin.append(GraphLinear(last_dim, in_dim))
        self.scale = paddle.create_parameter(
            paddle.zeros(shape=[1]).shape,
            paddle.zeros(shape=[1]).numpy().dtype,
            default_initializer=nn.initializer.Assign(paddle.zeros(shape=[1])),
        )

        mask = paddle.ones(shape=[n_node, in_dim])
        mask[masked_row, :] = 0
        self.register_buffer(name="mask", tensor=mask)

    def forward(self, adj, input):
        masked_x = self.mask * input
        s, t = self._s_t_function(adj, masked_x)
        if self.affine:
            out = masked_x + (1 - self.mask) * (input + t) * s
            logdet = paddle.sum(
                x=paddle.log(x=paddle.abs(x=s)).reshape([tuple(input.shape)[0], -1]),
                axis=1,
            )
        else:
            out = masked_x + t * (1 - self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x):
        s = None
        h = x
        for i in range(len(self.net)):
            h = self.net[i](adj, h)
            h = self.norm[i](h)
            h = nn.functional.relu(x=h)
        for i in range(len(self.net_lin) - 1):
            h = self.net_lin[i](h)
            h = self.norm_lin[i](h)
            h = nn.functional.relu(x=h)
        h = self.net_lin[-1](h)
        if self.affine:
            log_s, t = h.chunk(chunks=2, axis=-1)
            s = nn.functional.sigmoid(x=log_s)
        else:
            t = h
        return s, t


class Flow(nn.Layer):
    def __init__(
        self, in_channel, hidden_channels, affine=True, conv_lu=2, mask_swap=False
    ):
        super(Flow, self).__init__()
        self.actnorm = ActNorm(in_channel)
        if conv_lu == 0:
            self.invconv = InvConv2d(in_channel)
        elif conv_lu == 1:
            self.invconv = InvConv2dLU(in_channel)
        elif conv_lu == 2:
            self.invconv = None
        else:
            raise ValueError(
                "conv_lu in {0,1,2}, 0:InvConv2d, 1:InvConv2dLU, 2:none-just swap to update in coupling"
            )
        self.coupling = AffineCoupling(
            in_channel, hidden_channels, affine=affine, mask_swap=mask_swap
        )

    def forward(self, input):
        out, logdet = self.actnorm(input)
        if self.invconv:
            out, det1 = self.invconv(out)
        else:
            det1 = 0
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        if self.invconv:
            input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class FlowOnGraph(nn.Layer):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(FlowOnGraph, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine
        self.actnorm = ActNorm2D(in_dim=n_node)
        self.coupling = GraphAffineCoupling(
            n_node, in_dim, hidden_dim_dict, masked_row, affine=affine
        )

    def forward(self, adj, input):
        out, logdet = self.actnorm(input)
        det1 = 0
        out, det2 = self.coupling(adj, out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, adj, output):
        input = self.coupling.reverse(adj, output)
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Layer):
    def __init__(
        self, in_channel, n_flow, squeeze_fold, hidden_channels, affine=True, conv_lu=2
    ):
        super(Block, self).__init__()
        self.squeeze_fold = squeeze_fold
        squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold
        self.flows = nn.LayerList()
        for i in range(n_flow):
            if conv_lu in (0, 1):
                self.flows.append(
                    Flow(
                        squeeze_dim,
                        hidden_channels,
                        affine=affine,
                        conv_lu=conv_lu,
                        mask_swap=False,
                    )
                )
            else:
                self.flows.append(
                    Flow(
                        squeeze_dim,
                        hidden_channels,
                        affine=affine,
                        conv_lu=2,
                        mask_swap=bool(i % 2),
                    )
                )

    def forward(self, input):
        out = self._squeeze(input)
        logdet = 0
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det
        out = self._unsqueeze(out)
        return out, logdet

    def reverse(self, output):
        input = self._squeeze(output)
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        unsqueezed = self._unsqueeze(input)
        return unsqueezed

    def _squeeze(self, x):
        """Trade spatial extent for channels. In forward direction, convert each
        1x4x4 volume of input into a 4x1x1 volume of output.

        Args:
            x (paddle.Tensor): Input to squeeze or unsqueeze.
            reverse (bool): Reverse the operation, i.e., unsqueeze.

        Returns:
            x (paddle.Tensor): Squeezed or unsqueezed tensor.
        """
        assert len(tuple(x.shape)) == 4
        b_size, n_channel, height, width = tuple(x.shape)
        fold = self.squeeze_fold
        squeezed = x.reshape(
            [b_size, n_channel, height // fold, fold, width // fold, fold]
        )
        squeezed = squeezed.transpose(perm=[0, 1, 3, 5, 2, 4]).contiguous()
        out = squeezed.reshape(
            [b_size, n_channel * fold * fold, height // fold, width // fold]
        )
        return out

    def _unsqueeze(self, x):
        assert len(tuple(x.shape)) == 4
        b_size, n_channel, height, width = tuple(x.shape)
        fold = self.squeeze_fold
        unsqueezed = x.reshape(
            [b_size, n_channel // (fold * fold), fold, fold, height, width]
        )
        unsqueezed = unsqueezed.transpose(perm=[0, 1, 4, 2, 5, 3]).contiguous()
        out = unsqueezed.reshape(
            [b_size, n_channel // (fold * fold), height * fold, width * fold]
        )
        return out


class BlockOnGraph(nn.Layer):
    def __init__(
        self,
        n_node,
        in_dim,
        hidden_dim_dict,
        n_flow,
        mask_row_size=1,
        mask_row_stride=1,
        affine=True,
    ):
        """

        :param n_node:
        :param in_dim:
        :param hidden_dim:
        :param n_flow:
        :param mask_row_size: number of rows to be masked for update
        :param mask_row_stride: number of steps between two masks' firs row
        :param affine:
        """
        super(BlockOnGraph, self).__init__()
        assert 0 < mask_row_size < n_node
        self.flows = nn.LayerList()
        for i in range(n_flow):
            start = i * mask_row_stride
            masked_row = [(r % n_node) for r in range(start, start + mask_row_size)]
            self.flows.append(
                FlowOnGraph(
                    n_node,
                    in_dim,
                    hidden_dim_dict,
                    masked_row=masked_row,
                    affine=affine,
                )
            )

    def forward(self, adj, input):
        out = input
        logdet = 0
        for flow in self.flows:
            out, det = flow(adj, out)
            logdet = logdet + det
        return out, logdet

    def reverse(self, adj, output):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(adj, input)
        return input


class Glow(nn.Layer):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_block,
        squeeze_fold,
        hidden_channel,
        affine=True,
        conv_lu=2,
    ):
        super(Glow, self).__init__()
        self.blocks = nn.LayerList()
        n_channel = in_channel
        for i in range(n_block):
            self.blocks.append(
                Block(
                    n_channel,
                    n_flow,
                    squeeze_fold,
                    hidden_channel,
                    affine=affine,
                    conv_lu=conv_lu,
                )
            )

    def forward(self, input):
        logdet = 0
        out = input
        for block in self.blocks:
            out, det = block(out)
            logdet = logdet + det
        return out, logdet

    def reverse(self, z):
        h = z
        for i, block in enumerate(self.blocks[::-1]):
            h = block.reverse(h)
        return h


class GlowOnGraph(nn.Layer):
    def __init__(
        self,
        n_node,
        in_dim,
        hidden_dim_dict,
        n_flow,
        n_block,
        mask_row_size_list=[2],
        mask_row_stride_list=[1],
        affine=True,
    ):
        super(GlowOnGraph, self).__init__()
        assert len(mask_row_size_list) == n_block or len(mask_row_size_list) == 1
        assert len(mask_row_stride_list) == n_block or len(mask_row_stride_list) == 1
        if len(mask_row_size_list) == 1:
            mask_row_size_list = mask_row_size_list * n_block
        if len(mask_row_stride_list) == 1:
            mask_row_stride_list = mask_row_stride_list * n_block
        self.blocks = nn.LayerList()
        for i in range(n_block):
            mask_row_size = mask_row_size_list[i]
            mask_row_stride = mask_row_stride_list[i]
            self.blocks.append(
                BlockOnGraph(
                    n_node,
                    in_dim,
                    hidden_dim_dict,
                    n_flow,
                    mask_row_size,
                    mask_row_stride,
                    affine=affine,
                )
            )

    def forward(self, adj, x):
        logdet = 0
        out = x
        for block in self.blocks:
            out, det = block(adj, out)
            logdet = logdet + det
        return out, logdet

    def reverse(self, adj, z):
        input = z
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(adj, input)
        return input

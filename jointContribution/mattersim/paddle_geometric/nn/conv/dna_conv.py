import math
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.typing import Adj, OptTensor, SparseTensor


class Linear(Layer):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        # Define weight and bias parameters
        self.weight = self.create_parameter(
            shape=[groups, in_channels // groups, out_channels // groups])

        if bias:
            self.bias = self.create_parameter(shape=[out_channels])
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight and bias using Kaiming and uniform initialization
        paddle.nn.initializer.KaimingUniform()(self.weight)
        if self.bias is not None:
            fan_in = self.in_channels
            bound = 1 / math.sqrt(fan_in)
            paddle.nn.initializer.Uniform(-bound, bound)(self.bias)

    def forward(self, src):
        # Handles grouped linear transformation
        if self.groups > 1:
            size = src.shape[:-1]
            src = src.reshape((-1, self.groups, self.in_channels // self.groups))
            src = src.transpose((1, 0, 2))
            out = paddle.matmul(src, self.weight)
            out = out.transpose((1, 0, 2)).reshape(size + (self.out_channels,))
        else:
            out = paddle.matmul(src, self.weight.squeeze(0))

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, groups={self.groups})'


def restricted_softmax(src, dim: int = -1, margin: float = 0.):
    # Apply softmax with margin restriction
    src_max = paddle.clip(paddle.max(src, axis=dim, keepdim=True), min=0.)
    out = paddle.exp(src - src_max)
    out = out / (paddle.sum(out, axis=dim, keepdim=True) + paddle.exp(margin - src_max))
    return out


class Attention(Layer):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):
        # Computes attention using dot product
        score = paddle.matmul(query, key.transpose((-2, -1)))
        score = score / math.sqrt(key.shape[-1])
        score = restricted_softmax(score, axis=-1)
        score = F.dropout(score, p=self.dropout, training=self.training)
        return paddle.matmul(score, value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dropout={self.dropout})'


class MultiHead(Attention):
    def __init__(self, in_channels, out_channels, heads=1, groups=1, dropout=0, bias=True):
        super().__init__(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias

        # Ensure channels are compatible with the number of heads and groups
        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0

        # Define linear layers for query, key, and value
        self.lin_q = Linear(in_channels, out_channels, groups, bias)
        self.lin_k = Linear(in_channels, out_channels, groups, bias)
        self.lin_v = Linear(in_channels, out_channels, groups, bias)

        self.reset_parameters()

    def reset_parameters(self):
        # Reset parameters of linear layers
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value):
        # Applies multi-head attention over the query, key, and value tensors
        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        size = query.shape[:-2]
        out_channels_per_head = self.out_channels // self.heads

        # Reshape for multi-head attention
        query = query.reshape(size + (query.shape[-2], self.heads, out_channels_per_head)).transpose((-3, -2))
        key = key.reshape(size + (key.shape[-2], self.heads, out_channels_per_head)).transpose((-3, -2))
        value = value.reshape(size + (value.shape[-2], self.heads, out_channels_per_head)).transpose((-3, -2))

        out = self.compute_attention(query, key, value)
        out = out.transpose((-3, -2)).reshape(size + (query.shape[-2], self.out_channels))
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, '
                f'heads={self.heads}, groups={self.groups}, dropout={self.dropout}, bias={self.bias})')


class DNAConv(MessagePassing):
    def __init__(self, channels: int, heads: int = 1, groups: int = 1,
                 dropout: float = 0., cached: bool = False,
                 normalize: bool = True, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.bias = bias
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        # Initialize multi-head attention for DNA convolution
        self.multi_head = MultiHead(channels, channels, heads, groups, dropout, bias)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.multi_head.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        # Runs the forward pass, ensuring correct input shape
        if x.dim() != 3:
            raise ValueError('Feature shape must be [num_nodes, num_layers, channels].')

        # Normalize edge weights if required
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        # Applies multi-head attention to the messages
        x_i = x_i[:, -1:]  # [num_edges, 1, channels]
        out = self.multi_head(x_i, x_j, x_j)  # [num_edges, 1, channels]
        return edge_weight.view(-1, 1) * out.squeeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.multi_head.in_channels}, '
                f'heads={self.multi_head.heads}, '
                f'groups={self.multi_head.groups})')

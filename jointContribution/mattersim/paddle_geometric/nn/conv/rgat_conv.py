from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, Linear, ReLU

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import glorot, ones, zeros
from paddle_geometric.typing import Adj, OptTensor, Size
from paddle_geometric.utils import scatter, softmax


class RGATConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        mod: Optional[str] = None,
        attention_mechanism: str = "across-relation",
        attention_mode: str = "additive-self-attention",
        heads: int = 1,
        dim: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.mod = mod
        self.activation = ReLU()
        self.concat = concat
        self.attention_mode = attention_mode
        self.attention_mechanism = attention_mechanism
        self.dim = dim
        self.edge_dim = edge_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if self.attention_mechanism not in ["within-relation", "across-relation"]:
            raise ValueError('attention mechanism must either be '
                             '"within-relation" or "across-relation"')

        if self.attention_mode not in ["additive-self-attention", "multiplicative-self-attention"]:
            raise ValueError('attention mode must either be '
                             '"additive-self-attention" or "multiplicative-self-attention"')

        # Query and Key parameters
        self.q = paddle.create_parameter(
            shape=[heads * out_channels, heads * dim], dtype='float32')
        self.k = paddle.create_parameter(
            shape=[heads * out_channels, heads * dim], dtype='float32')

        # Bias handling
        if bias and concat:
            self.bias = paddle.create_parameter(
                shape=[heads * dim * out_channels], dtype='float32')
        elif bias and not concat:
            self.bias = paddle.create_parameter(
                shape=[dim * out_channels], dtype='float32')
        else:
            self.bias = None

        # Edge-specific linear transformation if edge_dim is provided
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias_attr=False)
            self.e = paddle.create_parameter(
                shape=[heads * out_channels, heads * dim], dtype='float32')
        else:
            self.lin_edge = None
            self.e = None

        # Basis handling if num_bases is provided
        if num_bases is not None:
            self.att = paddle.create_parameter(
                shape=[num_relations, num_bases], dtype='float32')
            self.basis = paddle.create_parameter(
                shape=[num_bases, in_channels, heads * out_channels], dtype='float32')
        # Block-wise handling if num_blocks is provided
        elif num_blocks is not None:
            assert (self.in_channels % num_blocks == 0) and (heads * out_channels) % num_blocks == 0, \
                ("Both 'in_channels' and 'heads * out_channels' must be "
                 "multiples of 'num_blocks' used.")
            self.weight = paddle.create_parameter(
                shape=[num_relations, num_blocks, self.in_channels // num_blocks,
                       (heads * out_channels) // num_blocks], dtype='float32')
        else:
            self.weight = paddle.create_parameter(
                shape=[num_relations, self.in_channels, heads * out_channels], dtype='float32')

        # Other weights
        self.w = paddle.create_parameter(shape=[out_channels], dtype='float32', default_initializer=paddle.nn.initializer.Constant(1))
        self.l1 = paddle.create_parameter(shape=[1, out_channels], dtype='float32')
        self.b1 = paddle.create_parameter(shape=[1, out_channels], dtype='float32')
        self.l2 = paddle.create_parameter(shape=[out_channels, out_channels], dtype='float32')
        self.b2 = paddle.create_parameter(shape=[1, out_channels], dtype='float32')


        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.num_bases is not None:
            glorot(self.basis)
            glorot(self.att)
        else:
            glorot(self.weight)
        glorot(self.q)
        glorot(self.k)
        zeros(self.bias)
        ones(self.l1)
        zeros(self.b1)
        self.l2.set_value(paddle.full(shape=self.l2.shape, fill_value=1 / self.out_channels))
        zeros(self.b2)
        if self.lin_edge is not None:
            glorot(self.lin_edge)
            glorot(self.e)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
    ):
        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x,
                             size=size, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.num_bases is not None:
            w = paddle.matmul(self.att, self.basis.reshape([self.num_bases, -1]))
            w = w.reshape([self.num_relations, self.in_channels, self.heads * self.out_channels])
        if self.num_blocks is not None:
            w = self.weight
            x_i = x_i.reshape([-1, 1, w.shape[1], w.shape[2]])
            x_j = x_j.reshape([-1, 1, w.shape[1], w.shape[2]])
            w = paddle.index_select(w, index=edge_type, axis=0)
            outi = paddle.einsum('abcd,acde->ace', x_i, w).reshape([-1, self.heads * self.out_channels])
            outj = paddle.einsum('abcd,acde->ace', x_j, w).reshape([-1, self.heads * self.out_channels])
        else:
            w = paddle.index_select(self.weight, index=edge_type, axis=0)
            outi = paddle.bmm(x_i.unsqueeze(1), w).squeeze(-2)
            outj = paddle.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        qi = paddle.matmul(outi, self.q)
        kj = paddle.matmul(outj, self.k)

        alpha_edge, alpha = 0, paddle.zeros([1])
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attributes = self.lin_edge(edge_attr).reshape([-1, self.heads * self.out_channels])
            if edge_attributes.shape[0] != edge_attr.shape[0]:
                edge_attributes = paddle.index_select(edge_attributes, index=edge_type, axis=0)
            alpha_edge = paddle.matmul(edge_attributes, self.e)

        if self.attention_mode == "additive-self-attention":
            alpha = F.leaky_relu(qi + kj + alpha_edge, self.negative_slope) if edge_attr is not None else F.leaky_relu(qi + kj, self.negative_slope)
        elif self.attention_mode == "multiplicative-self-attention":
            alpha = (qi * kj * alpha_edge) if edge_attr is not None else (qi * kj)

        if self.attention_mechanism == "within-relation":
            across_out = paddle.zeroslike(alpha)
            for r in range(self.num_relations):
                mask = edge_type == r
                across_out[mask] = softmax(alpha[mask], index[mask])
            alpha = across_out
        elif self.attention_mechanism == "across-relation":
            alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha

        if self.mod == "additive":
            return (outj.reshape([-1, self.heads, self.out_channels]) * alpha.unsqueeze(-1))

        elif self.mod == "scaled":
            degree = scatter(paddle.ones_like(alpha), index, dim_size=size_i, reduce='sum')[index].unsqueeze(-1)
            degree = paddle.matmul(degree, self.l1) + self.b1
            degree = self.activation(degree)
            degree = paddle.matmul(degree, self.l2) + self.b2
            return paddle.multiply(outj.reshape([-1, self.heads, self.out_channels]) * alpha.unsqueeze(-1), degree)

        return outj.reshape([-1, self.heads, self.out_channels]) * alpha.unsqueeze(-1)

    def update(self, aggr_out: Tensor) -> Tensor:
        aggr_out = aggr_out.reshape([-1, self.heads * self.dim * self.out_channels]) if self.concat else aggr_out.mean(axis=1)
        if self.bias is not None:
            aggr_out += self.bias
        return aggr_out

    def __repr__(self) -> str:
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

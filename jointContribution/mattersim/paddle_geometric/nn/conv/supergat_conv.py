import math
from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, Linear
from paddle.nn.initializer import XavierUniform, Constant

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.nn.inits import glorot, zeros
from paddle_geometric.typing import Adj, OptTensor, SparseTensor, paddle_sparse
from paddle_geometric.utils import (
    add_self_loops,
    batched_negative_sampling,
    dropout_edge,
    is_undirected,
    negative_sampling,
    remove_self_loops,
    softmax,
    to_undirected,
)

class SuperGATConv(MessagePassing):
    r"""The self-supervised graph attentional operator from the `"How to Find
    Your Friendly Neighborhood: Graph Attention Design with Self-Supervision"
    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        attention_type (str, optional): Type of attention to use
            (:obj:`'MX'`, :obj:`'SD'`). (default: :obj:`'MX'`)
        neg_sample_ratio (float, optional): The ratio of the number of sampled
            negative edges to the number of positive edges.
            (default: :obj:`0.5`)
        edge_sample_ratio (float, optional): The ratio of samples to use for
            training among the number of training edges. (default: :obj:`1.0`)
        is_undirected (bool, optional): Whether the input graph is undirected.
            If not given, will be automatically computed with the input graph
            when negative sampling is performed. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = True,
                 bias: bool = True, attention_type: str = 'MX',
                 neg_sample_ratio: float = 0.5, edge_sample_ratio: float = 1.0,
                 is_undirected: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.is_undirected = is_undirected

        assert attention_type in ['MX', 'SD']
        assert 0.0 < neg_sample_ratio and 0.0 < edge_sample_ratio <= 1.0

        self.lin = Linear(in_channels, heads * out_channels, bias_attr=False)
        self._glorot_initializer(self.lin.weight)

        if self.attention_type == 'MX':
            self.att_l = self.create_parameter(
                shape=(1, heads, out_channels),
                default_initializer=XavierUniform()
            )
            self.att_r = self.create_parameter(
                shape=(1, heads, out_channels),
                default_initializer=XavierUniform()
            )
        else:  # self.attention_type == 'SD'
            self.att_l = None
            self.att_r = None

        self.att_x = self.att_y = None  # x/y for self-supervision

        if bias and concat:
            self.bias = self.create_parameter(
                shape=(heads * out_channels,),
                default_initializer=Constant(0.0)
            )
        elif bias and not concat:
            self.bias = self.create_parameter(
                shape=(out_channels,),
                default_initializer=Constant(0.0)
            )
        else:
            self.bias = None

        self.reset_parameters()
    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._glorot_initializer(self.att_l)
        self._glorot_initializer(self.att_r)
        self._zeros_initializer(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        neg_edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        N, H, C = x.shape[0], self.heads, self.out_channels

        if self.add_self_loops:
            edge_index = add_self_loops(edge_index, num_nodes=N)

        x = self.lin(x).reshape([-1, H, C])

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x)

        if self.training:
            pos_edge_index = self.positive_sampling(edge_index)

            pos_att = self.get_attention(
                edge_index_i=pos_edge_index[1],
                x_i=x[pos_edge_index[1]],
                x_j=x[pos_edge_index[0]],
                num_nodes=x.shape[0],
                return_logits=True,
            )

            if neg_edge_index is None:
                neg_edge_index = self.negative_sampling(edge_index, N, batch)

            neg_att = self.get_attention(
                edge_index_i=neg_edge_index[1],
                x_i=x[neg_edge_index[1]],
                x_j=x[neg_edge_index[0]],
                num_nodes=x.shape[0],
                return_logits=True,
            )

            self.att_x = paddle.concat([pos_att, neg_att], axis=0)
            self.att_y = paddle.zeros_like(self.att_x)
            self.att_y[:pos_edge_index.shape[1]] = 1.

        if self.concat:
            out = out.reshape([-1, self.heads * self.out_channels])
        else:
            out = paddle.mean(out, axis=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.get_attention(edge_index_i, x_i, x_j, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def negative_sampling(self, edge_index: Tensor, num_nodes: int,
                          batch: Optional[Tensor] = None) -> Tensor:
        num_neg_samples = int(self.neg_sample_ratio * self.edge_sample_ratio *
                              edge_index.shape[1])

        if not self.is_undirected and not is_undirected(
                edge_index, num_nodes=num_nodes):
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        if batch is None:
            neg_edge_index = negative_sampling(edge_index, num_nodes,
                                               num_neg_samples=num_neg_samples)
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index, batch, num_neg_samples=num_neg_samples)

        return neg_edge_index

    def positive_sampling(self, edge_index: Tensor) -> Tensor:
        pos_edge_index, _ = dropout_edge(edge_index,
                                         p=1. - self.edge_sample_ratio,
                                         training=self.training)
        return pos_edge_index

    def get_attention(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                      num_nodes: Optional[int],
                      return_logits: bool = False) -> Tensor:

        if self.attention_type == 'MX':
            logits = paddle.sum(x_i * x_j, axis=-1)
            if return_logits:
                return logits

            alpha = paddle.sum(x_j * self.att_l, axis=-1) + paddle.sum(x_i * self.att_r, axis=-1)
            alpha = alpha * F.sigmoid(logits)

        else:  # self.attention_type == 'SD'
            alpha = paddle.sum(x_i * x_j, axis=-1) / math.sqrt(self.out_channels)
            if return_logits:
                return alpha

        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha

    def get_attention_loss(self) -> Tensor:
        r"""Computes the self-supervised graph attention loss."""
        if not self.training:
            return paddle.to_tensor([0], dtype=paddle.float32)

        return F.binary_cross_entropy_with_logits(
            paddle.mean(self.att_x, axis=-1),
            self.att_y,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'type={self.attention_type})')

    def _glorot_initializer(self, param):
        if param is not None:
            paddle.nn.initializer.XavierUniform()(param)

    def _zeros_initializer(self, param):
        if param is not None:
            paddle.assign(paddle.zeros_like(param), param)
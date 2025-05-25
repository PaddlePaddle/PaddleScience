from typing import Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense import Linear
from paddle_geometric.nn.inits import glorot, reset
from paddle_geometric.typing import PairTensor  # noqa
from paddle_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from paddle_geometric.utils import softmax


def group(
    xs: List[Tensor],
    q,
    k_lin: nn.Layer,
) -> Tuple[OptTensor, OptTensor]:
    if len(xs) == 0:
        return None, None
    else:
        num_edge_types = len(xs)
        out = paddle.stack(xs)
        if out.numel() == 0:
            return out.reshape([0, out.shape[-1]]), None
        attn_score = (q * paddle.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, axis=0)
        out = paddle.sum(attn.reshape([num_edge_types, 1, -1]) * out, axis=0)
        return out, attn


class HANConv(MessagePassing):
    r"""The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/abs/1903.07293>`_ paper.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = self.create_parameter(shape=[1, out_channels], default_initializer=nn.initializer.XavierUniform())

        self.proj = nn.LayerDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = self.create_parameter(shape=[1, heads, dim], default_initializer=nn.initializer.XavierUniform())
            self.lin_dst[edge_type] = self.create_parameter(shape=[1, heads, dim], default_initializer=nn.initializer.XavierUniform())

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.weight.set_value(paddle.nn.initializer.XavierUniform())
        self.q.set_value(paddle.nn.initializer.XavierUniform())

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        return_semantic_attention_weights: bool = False,
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor], Dict[NodeType, OptTensor]]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).reshape([-1, H, D])
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_src * lin_src).sum(axis=-1)
            alpha_dst = (x_dst * lin_dst).sum(axis=-1)
            out = self.propagate(edge_index, x=(x_src, x_dst), alpha=(alpha_src, alpha_dst))

            out = F.relu(out)
            out_dict[dst_type].append(out)

        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = group(outs, self.q, self.k_lin)
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return out_dict, semantic_attn_dict

        return out_dict

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor, index: Tensor, ptr: Optional[Tensor], size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.reshape([-1, self.heads, 1])
        return out.reshape([-1, self.out_channels])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels}, heads={self.heads})'

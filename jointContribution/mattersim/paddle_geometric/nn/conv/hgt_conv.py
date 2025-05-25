import math
from typing import Dict, List, Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer, LayerDict

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from paddle_geometric.nn.parameter_dict import ParameterDict
from paddle_geometric.typing import Adj, EdgeType, Metadata, NodeType
from paddle_geometric.utils import softmax
from paddle_geometric.utils.hetero import construct_bipartite_edge_index


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper."""

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(1.0))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = self.create_parameter(shape=[1, heads], default_initializer=paddle.nn.initializer.Constant(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        for key in self.skip.keys():
            self.skip[key].set_value(paddle.ones([1]))
        for key in self.p_rel.keys():
            self.p_rel[key].set_value(paddle.ones([1, self.heads]))

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.shape[0]
        return paddle.concat(outs, axis=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].shape[0]
            offset[edge_type] = cumsum
            cumsum += N

            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = paddle.arange(H, dtype='int64').unsqueeze(-1).tile([1, N]) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = paddle.concat(ks, axis=0).transpose([1, 0]).reshape([-1, D])
        vs = paddle.concat(vs, axis=0).transpose([1, 0]).reshape([-1, D])
        type_vec = paddle.concat(type_list, axis=1).flatten()

        k = self.k_rel(ks, type_vec).reshape([H, -1, D]).transpose([1, 0, 2])
        v = self.v_rel(vs, type_vec).reshape([H, -1, D]).transpose([1, 0, 2])

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = paddle.split(val, 3, axis=1)
            k_dict[key] = k.reshape([-1, H, D])
            q_dict[key] = q.reshape([-1, H, D])
            v_dict[key] = v.reshape([-1, H, D])

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.shape[0])

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].shape[0]
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        a_dict = self.out_lin({
            k:
            paddle.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.shape[-1] == x_dict[node_type].shape[-1]:
                alpha = paddle.nn.functional.sigmoid(self.skip[node_type])
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(axis=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.shape[-1])
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.unsqueeze(-1)
        return out.reshape([-1, self.out_channels])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')

from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric import EdgeIndex
from paddle_geometric.nn.conv.cugraph import CuGraphModule
from paddle_geometric.nn.conv.cugraph.base import LEGACY_MODE
from paddle_geometric.nn.inits import glorot, zeros

try:
    if LEGACY_MODE:
        from paddlenlp.ops.torch.autograd import \
            agg_hg_basis_n2n_post as RGCNConvAgg
    else:
        from paddlenlp.ops.paddle.operators import \
            agg_hg_basis_n2n_post as RGCNConvAgg
except ImportError:
    pass


class CuGraphRGCNConv(CuGraphModule):  # pragma: no cover
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.

    :class:`CuGraphRGCNConv` is an optimized version of
    :class:`~torch_geometric.nn.conv.RGCNConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """
    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 num_bases: Optional[int] = None, aggr: str = 'mean',
                 root_weight: bool = True, bias: bool = True):
        super().__init__()

        if aggr not in ['sum', 'add', 'mean']:
            raise ValueError(f"Aggregation function must be either 'mean' "
                             f"or 'sum' (got '{aggr}')")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.aggr = aggr
        self.root_weight = root_weight

        dim_root_weight = 1 if root_weight else 0

        if num_bases is not None:
            self.weight = paddle.create_parameter(
                shape=[num_bases + dim_root_weight, in_channels, out_channels],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform())
            self.comp = paddle.create_parameter(
                shape=[num_relations, num_bases],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform())
        else:
            self.weight = paddle.create_parameter(
                shape=[num_relations + dim_root_weight, in_channels, out_channels],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform())
            self.comp = None  # Register comp as None if no num_bases

        if bias:
            self.bias = paddle.create_parameter(
                shape=[out_channels],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform())
        else:
            self.bias = None  # Register bias as None if bias is False

        self.reset_parameters()

    def reset_parameters(self):
        end = -1 if self.root_weight else None
        glorot(self.weight[:end])
        glorot(self.comp)
        if self.root_weight:
            glorot(self.weight[-1])
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeIndex,
        edge_type: Tensor,
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            edge_index (EdgeIndex): The edge indices.
            edge_type (torch.Tensor): The edge type.
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph.. When not given, the value will be computed
                on-the-fly, leading to slightly worse performance.
                (default: :obj:`None`)
        """
        graph = self.get_typed_cugraph(edge_index, edge_type,
                                       self.num_relations, max_num_neighbors)

        out = RGCNConvAgg(x, self.comp, graph, concat_own=self.root_weight,
                          norm_by_out_degree=bool(self.aggr == 'mean'))

        out = out @ self.weight.view(-1, self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')

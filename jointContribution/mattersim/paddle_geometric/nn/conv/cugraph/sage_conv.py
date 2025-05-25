from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Linear

from paddle_geometric import EdgeIndex
from paddle_geometric.nn.conv.cugraph import CuGraphModule
from paddle_geometric.nn.conv.cugraph.base import LEGACY_MODE

try:
    if LEGACY_MODE:
        from pylibcugraphops.paddle.autograd import \
            agg_concat_n2n as SAGEConvAgg
    else:
        from pylibcugraphops.paddle.operators import \
            agg_concat_n2n as SAGEConvAgg
except ImportError:
    pass


class CuGraphSAGEConv(CuGraphModule):  # pragma: no cover
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    :class:`CuGraphSAGEConv` is an optimized version of
    :class:`~paddle_geometric.nn.conv.SAGEConv` based on the :obj:`cugraph-ops`
    package that fuses message passing computation for accelerated execution
    and lower memory footprint.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        if aggr not in ['mean', 'sum', 'min', 'max']:
            raise ValueError(f"Aggregation function must be either 'mean', "
                             f"'sum', 'min' or 'max' (got '{aggr}')")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if self.project:
            self.pre_lin = Linear(in_channels, in_channels, bias_attr=True)

        if self.root_weight:
            self.lin = Linear(2 * in_channels, out_channels, bias_attr=bias)
        else:
            self.lin = Linear(in_channels, out_channels, bias_attr=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.pre_lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: EdgeIndex,
        max_num_neighbors: Optional[int] = None,
    ) -> Tensor:
        graph = self.get_cugraph(edge_index, max_num_neighbors)

        if self.project:
            x = F.relu(self.pre_lin(x))

        out = SAGEConvAgg(x, graph, self.aggr)

        if self.root_weight:
            out = self.lin(out)
        else:
            out = self.lin(out[:, :self.in_channels])

        if self.normalize:
            out = F.normalize(out, p=2., axis=-1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

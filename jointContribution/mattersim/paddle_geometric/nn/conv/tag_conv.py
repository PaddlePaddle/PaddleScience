import warnings
from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.nn.inits import zeros
from paddle_geometric.typing import Adj, OptTensor, SparseTensor
from paddle_geometric.utils import spmm
from paddle_geometric.nn.conv.gcn_conv import gcn_norm


class TAGConv(MessagePassing):
    r"""The topology adaptive graph convolutional networks operator from the
    `"Topology Adaptive Graph Convolutional Networks"
    <https://arxiv.org/abs/1710.10370>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^K \left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)^k \mathbf{X} \mathbf{W}_{k},

    where :math:`\mathbf{A}` denotes the adjacency matrix and
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`3`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to apply symmetric normalization.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node_features :math:`(|\mathcal{V}|, F_{in})`,
          edge_index :math:`(2, |\mathcal{E}|)`,
          edge_weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_h: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, K: int = 3,
                 bias: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize

        self.lins = paddle.nn.LayerList([
            Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)
        ])

        if bias:
            self.bias = paddle.nn.Parameter(paddle.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_h
        if cache is None:
            if isinstance(edge_index, paddle.Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.shape[0], False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.shape[0], False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            h = x * self.alpha
            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                h = h + (1 - self.alpha) / self.K * x
            if self.cached:
                self._cached_h = h
        else:
            h = cache.detach()

        return self.lin(h)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

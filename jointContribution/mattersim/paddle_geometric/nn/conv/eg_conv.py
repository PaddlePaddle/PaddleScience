from typing import List, Optional, Tuple

import paddle
from paddle import Tensor
from paddle.nn import Layer, Linear
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.nn.inits import zeros

from paddle_geometric.typing import Adj, OptTensor, SparseTensor
from paddle_geometric.utils import add_remaining_self_loops, scatter, spmm


class EGConv(MessagePassing):
    r"""The Efficient Graph Convolution from the `"Adaptive Filters and
    Aggregator Fusion for Efficient Graph Convolutions"
    <https://arxiv.org/abs/2104.01481>`_ paper.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}_i^{\prime} = {\LARGE ||}_{h=1}^H \sum_{\oplus \in
        \mathcal{A}} \sum_{b = 1}^B w_{i, h, \oplus, b} \;
        \underset{j \in \mathcal{N}(i) \cup \{i\}}{\bigoplus}
        \mathbf{W}_b \mathbf{x}_{j}

    with :math:`\mathbf{W}_b` denoting a basis weight,
    :math:`\oplus` denoting an aggregator, and :math:`w` denoting per-vertex
    weighting coefficients across different heads, bases and aggregators.

    EGC retains :math:`\mathcal{O}(|\mathcal{V}|)` memory usage, making it a
    sensible alternative to :class:`~paddle_geometric.nn.conv.GCNConv`,
    :class:`~paddle_geometric.nn.conv.SAGEConv` or
    :class:`~paddle_geometric.nn.conv.GINConv`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (List[str], optional): Aggregators to be used.
            Supported aggregators are :obj:`"sum"`, :obj:`"mean"`,
            :obj:`"symnorm"`, :obj:`"max"`, :obj:`"min"`, :obj:`"std"`,
            :obj:`"var"`.
            Multiple aggregators can be used to improve the performance.
            (default: :obj:`["symnorm"]`)
        num_heads (int, optional): Number of heads :math:`H` to use. Must have
            :obj:`out_channels % num_heads == 0`. It is recommended to set
            :obj:`num_heads >= num_bases`. (default: :obj:`8`)
        num_bases (int, optional): Number of basis weights :math:`B` to use.
            (default: :obj:`4`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of the edge index with added self loops on first
            execution, along with caching the calculation of the symmetric
            normalized edge weights if the :obj:`"symnorm"` aggregator is
            being used. This parameter should only be set to :obj:`True` in
            transductive learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str] = ['symnorm'],
        num_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        if out_channels % num_heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads "
                             f"(got {num_heads})")

        for a in aggregators:
            if a not in ['sum', 'mean', 'symnorm', 'min', 'max', 'var', 'std']:
                raise ValueError(f"Unsupported aggregator: '{a}'")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = aggregators

        self.bases_lin = Linear(in_channels,
                                (out_channels // num_heads) * num_bases,
                                bias_attr=False)
        self.comb_lin = Linear(in_channels,
                               num_heads * num_bases * len(aggregators))

        if bias:
            self.bias = self.create_parameter(shape=[out_channels])
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        self.bases_lin.reset_parameters()
        self.comb_lin.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators:
            edge_index, symnorm_weight = gcn_norm(
                edge_index, None, num_nodes=x.shape[0],
                add_self_loops=self.add_self_loops, dtype=x.dtype)

        elif self.add_self_loops:
            edge_index, _ = add_remaining_self_loops(edge_index)

        bases = self.bases_lin(x)
        weightings = self.comb_lin(x)

        aggregated = self.propagate(edge_index, x=bases, symnorm_weight=symnorm_weight)

        weightings = weightings.reshape([-1, self.num_heads, self.num_bases * len(self.aggregators)])
        aggregated = aggregated.reshape(
            [-1, len(self.aggregators) * self.num_bases, self.out_channels // self.num_heads])

        out = paddle.matmul(weightings, aggregated)
        out = out.reshape([-1, self.out_channels])

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None,
                  symnorm_weight: OptTensor = None) -> Tensor:

        outs = []
        for aggr in self.aggregators:
            if aggr == 'symnorm':
                out = scatter(inputs * symnorm_weight, index, axis=0, dim_size=dim_size, reduce='sum')
            elif aggr == 'var' or aggr == 'std':
                mean = scatter(inputs, index, axis=0, dim_size=dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, axis=0, dim_size=dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggr == 'std':
                    out = paddle.sqrt(out.clip(min=1e-5))
            else:
                out = scatter(inputs, index, axis=0, dim_size=dim_size, reduce=aggr)

            outs.append(out)

        return paddle.stack(outs, axis=1) if len(outs) > 1 else outs[0]

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        adj_t_2 = adj_t
        if len(self.aggregators) > 1 and 'symnorm' in self.aggregators:
            adj_t_2 = adj_t.set_value(None) if isinstance(adj_t, SparseTensor) else adj_t.clone().fill_diagonal(1.0)

        outs = []
        for aggr in self.aggregators:
            if aggr == 'symnorm':
                out = spmm(adj_t, x, reduce='sum')
            elif aggr in ['var', 'std']:
                mean = spmm(adj_t_2, x, reduce='mean')
                mean_sq = spmm(adj_t_2, x * x, reduce='mean')
                out = mean_sq - mean * mean
                if aggr == 'std':
                    out = paddle.sqrt(out.clip(min=1e-5))
            else:
                out = spmm(adj_t_2, x, reduce=aggr)

            outs.append(out)

        return paddle.stack(outs, axis=1) if len(outs) > 1 else outs[0]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggregators={self.aggregators})')

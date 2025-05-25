from typing import Optional, Tuple

import paddle
from paddle import Tensor
from paddle.nn import Layer, Linear
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import spmm


class PANConv(MessagePassing):
    r"""The path integral based convolutional operator from the
    `"Path Integral Based Convolution and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2006.16811>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{M} \mathbf{X} \mathbf{W}

    where :math:`\mathbf{M}` denotes the normalized and learned maximal entropy
    transition (MET) matrix that includes neighbors up to :obj:`filter_size`
    hops:

    .. math::

        \mathbf{M} = \mathbf{Z}^{-1/2} \sum_{n=0}^L e^{-\frac{E(n)}{T}}
        \mathbf{A}^n \mathbf{Z}^{-1/2}

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        filter_size (int): The filter size :math:`L`.
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int, filter_size: int,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        self.lin = Linear(in_channels, out_channels)
        self.weight = self.create_parameter(shape=[filter_size + 1], default_initializer=paddle.nn.initializer.Constant(0.5))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.weight.set_value(paddle.full([self.filter_size + 1], 0.5))

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
    ) -> Tuple[Tensor, SparseTensor]:

        adj_t: Optional[SparseTensor] = None
        if isinstance(edge_index, Tensor):
            adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                                 sparse_sizes=(x.shape[0], x.shape[0]))
        elif isinstance(edge_index, SparseTensor):
            adj_t = edge_index.set_value(None)

        adj_t = self.panentropy(adj_t, dtype=x.dtype)

        deg = adj_t.storage.rowcount().astype(x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = paddle.where(deg_inv_sqrt == float('inf'), paddle.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        M = deg_inv_sqrt.reshape([1, -1]) * adj_t * deg_inv_sqrt.reshape([-1, 1])

        out = self.propagate(M, x=x, edge_weight=None)
        out = self.lin(out)
        return out, M

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.reshape([-1, 1]) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def panentropy(self, adj_t: SparseTensor,
                   dtype: Optional[str] = None) -> SparseTensor:

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0)

        tmp = SparseTensor.eye(adj_t.shape[0], adj_t.shape[1], has_value=True,
                               dtype=dtype, device=adj_t.place)
        tmp = tmp.mul_nnz(self.weight[0])

        outs = [tmp]
        for i in range(1, self.filter_size + 1):
            tmp = tmp @ adj_t
            tmp = tmp.mul_nnz(self.weight[i])
            outs.append(tmp)

        row = paddle.concat([out.storage.row() for out in outs], axis=0)
        col = paddle.concat([out.storage.col() for out in outs], axis=0)
        value = paddle.concat([out.storage.value() for out in outs], axis=0)

        out = SparseTensor(row=row, col=col, value=value,
                           sparse_sizes=adj_t.sparse_sizes()).coalesce()

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, filter_size={self.filter_size})')

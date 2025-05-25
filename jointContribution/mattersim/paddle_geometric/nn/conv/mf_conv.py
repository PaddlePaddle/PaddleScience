from typing import Tuple, Union

from paddle import Tensor
from paddle.nn import LayerList

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from paddle_geometric.utils import degree, spmm


class MFConv(MessagePassing):
    r"""The graph neural network operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}^{(\deg(i))}_1 \mathbf{x}_i +
        \mathbf{W}^{(\deg(i))}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j

    which trains a distinct weight matrix for each possible vertex degree.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        max_degree (int, optional): The maximum node degree to consider when
            updating weights (default: :obj:`10`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, max_degree: int = 10, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lins_l = LayerList([
            Linear(in_channels[0], out_channels, bias_attr=bias)
            for _ in range(max_degree + 1)
        ])

        self.lins_r = LayerList([
            Linear(in_channels[1], out_channels, bias_attr=False)
            for _ in range(max_degree + 1)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins_l:
            lin.reset_parameters()
        for lin in self.lins_r:
            lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        x_r = x[1]

        # Compute degree of each node
        if isinstance(edge_index, SparseTensor):
            deg = edge_index.storage.row_count()
        elif isinstance(edge_index, Tensor):
            i = 1 if self.flow == 'source_to_target' else 0
            N = x[0].shape[self.node_dim]
            N = size[1] if size is not None else N
            N = x_r.shape[self.node_dim] if x_r is not None else N
            deg = degree(edge_index[i], N, dtype='int64')
        deg = paddle.clip(deg, max=self.max_degree)

        # propagate_type: (x: OptPairTensor)
        h = self.propagate(edge_index, x=x, size=size)

        out = paddle.zeros(shape=(h.shape[0], self.out_channels), dtype=h.dtype)
        for i, (lin_l, lin_r) in enumerate(zip(self.lins_l, self.lins_r)):
            idx = paddle.nonzero(deg == i).flatten()
            r = lin_l(h.index_select(idx, axis=self.node_dim))

            if x_r is not None:
                r = r + lin_r(x_r.index_select(idx, axis=self.node_dim))

            out.scatter_(idx, r, overwrite=True)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x[0], reduce=self.aggr)

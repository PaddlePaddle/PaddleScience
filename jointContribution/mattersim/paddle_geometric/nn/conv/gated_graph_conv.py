import paddle
from paddle import Tensor
from paddle.nn import Layer, GRUCell, Linear
from paddle.nn.initializer import Uniform

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.typing import Adj, OptTensor
from paddle_geometric.utils import spmm


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the
    `"Gated Graph Sequence Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper.

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    Args:
        out_channels (int): Size of each output sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = self.create_parameter(
            shape=[num_layers, out_channels, out_channels],
            default_initializer=Uniform()
        )
        self.rnn = GRUCell(input_size=out_channels, hidden_size=out_channels, bias_attr=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_value(paddle.uniform(self.weight.shape, min=-1.0, max=1.0))
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if x.shape[-1] > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.shape[-1] < self.out_channels:
            zero = paddle.zeros([x.shape[0], self.out_channels - x.shape[-1]], dtype=x.dtype)
            x = paddle.concat([x, zero], axis=1)

        for i in range(self.num_layers):
            m = paddle.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.unsqueeze(-1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels}, num_layers={self.num_layers})'

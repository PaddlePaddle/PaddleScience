from typing import Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer, Linear

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import normal
from paddle_geometric.typing import Adj, PairTensor, SparseTensor
from paddle_geometric.utils import add_self_loops, remove_self_loops


class FeaStConv(MessagePassing):
    r"""The (translation-invariant) feature-steered convolutional operator from
    the `"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis"
    <https://arxiv.org/abs/1706.05206>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \sum_{h=1}^H
        q_h(\mathbf{x}_i, \mathbf{x}_j) \mathbf{W}_h \mathbf{x}_j

    with :math:`q_h(\mathbf{x}_i, \mathbf{x}_j) = \mathrm{softmax}_j
    (\mathbf{u}_h^{\top} (\mathbf{x}_j - \mathbf{x}_i) + c_h)`, where :math:`H`
    denotes the number of attention heads, and :math:`\mathbf{W}_h`,
    :math:`\mathbf{u}_h` and :math:`c_h` are trainable parameters.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of attention heads :math:`H`.
            (default: :obj:`1`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, heads * out_channels, bias_attr=False)
        self.u = Linear(in_channels, heads, bias_attr=False)
        self.c = self.create_parameter(shape=[heads], default_initializer=normal(0, 0.1))

        if bias:
            self.bias = self.create_parameter(shape=[out_channels], default_initializer=normal(0, 0.1))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.weight.set_value(normal(0, 0.1)(self.lin.weight.shape))
        self.u.weight.set_value(normal(0, 0.1)(self.u.weight.shape))
        self.c.set_value(normal(0, 0.1)(self.c.shape))
        if self.bias is not None:
            self.bias.set_value(normal(0, 0.1)(self.bias.shape))

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].shape[0])
            elif isinstance(edge_index, SparseTensor):
                edge_index = edge_index + paddle.sparse.eye(edge_index.shape[0], edge_index.shape[1])

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        q = self.u(x_j - x_i) + self.c  # Translation invariance.
        q = F.softmax(q, axis=1)
        x_j = self.lin(x_j).reshape([x_j.shape[0], self.heads, -1])
        return (x_j * q.unsqueeze(-1)).sum(axis=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

from typing import Union, Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Linear

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.typing import Adj, PairTensor, SparseTensor
from paddle_geometric.utils import spmm


class SignedConv(MessagePassing):
    r"""
    The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.

    This operator computes node embeddings using positive and negative edges
    as described in the paper. It has two different aggregation modes:

    1. If `first_aggr` is set to `True`, positive and negative embeddings are
       computed using separate transformations for each, and then combined.
    2. If `first_aggr` is set to `False`, the input features are expected to
       be concatenated for positive and negative node features.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to `False`, the layer will not learn an
            additive bias. (default: `True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          positive edge indices :math:`(2, |\mathcal{E}^{(+)}|)`,
          negative edge indices :math:`(2, |\mathcal{E}^{(-)}|)`
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):
        """
        Initialize the SignedConv layer.
        """
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos_l = Linear(in_channels, out_channels, bias_attr=False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias_attr=bias)
            self.lin_neg_l = Linear(in_channels, out_channels, bias_attr=False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias_attr=bias)
        else:
            self.lin_pos_l = Linear(2 * in_channels, out_channels, bias_attr=False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias_attr=bias)
            self.lin_neg_l = Linear(2 * in_channels, out_channels, bias_attr=False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias_attr=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the SignedConv layer.
        """
        super().reset_parameters()
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos_edge_index: Adj,
        neg_edge_index: Adj,
    ):
        """
        Forward pass of the SignedConv layer.
        """
        if isinstance(x, Tensor):
            x = (x, x)

        if self.first_aggr:
            # Aggregating positive and negative edge information separately
            out_pos = self.propagate(pos_edge_index, x=x)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1])

            out_neg = self.propagate(neg_edge_index, x=x)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1])

            return paddle.concat([out_pos, out_neg], axis=-1)
        else:
            F_in = self.in_channels

            # Aggregating with concatenated positive and negative features
            out_pos1 = self.propagate(pos_edge_index,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_pos2 = self.propagate(neg_edge_index,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_pos = paddle.concat([out_pos1, out_pos2], axis=-1)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1][..., :F_in])

            out_neg1 = self.propagate(pos_edge_index,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_neg2 = self.propagate(neg_edge_index,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_neg = paddle.concat([out_neg1, out_neg2], axis=-1)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1][..., F_in:])

            return paddle.concat([out_pos, out_neg], axis=-1)

    def message(self, x_j: Tensor) -> Tensor:
        """
        Compute the message to pass during the aggregation step.
        """
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: PairTensor) -> Tensor:
        """
        Message aggregation step.
        """
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        """
        String representation of the SignedConv layer.
        """
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr})')

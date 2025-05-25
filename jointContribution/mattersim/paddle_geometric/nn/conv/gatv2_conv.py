import typing
from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense.linear import Linear
from paddle_geometric.nn.inits import glorot, zeros
from paddle_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    paddle_sparse,
)
from paddle_geometric.utils import (
    add_self_loops,
    is_paddle_sparse_tensor,
    remove_self_loops,
    softmax,
)
from paddle_geometric.utils.sparse import set_sparse_value

from typing import overload

class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_j
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_k
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}])
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or paddle.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`paddle.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:`\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}`.
            (default: :obj:`False`)
        residual (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V_t}|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = paddle.nn.Linear(in_channels, heads * out_channels, bias_attr=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = paddle.nn.Linear(in_channels, heads * out_channels, bias_attr=bias)
        else:
            self.lin_l = paddle.nn.Linear(in_channels[0], heads * out_channels, bias_attr=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = paddle.nn.Linear(in_channels[1], heads * out_channels, bias_attr=bias)

        self.att = self.create_parameter([1, heads, out_channels])

        if edge_dim is not None:
            self.lin_edge = paddle.nn.Linear(edge_dim, heads * out_channels, bias_attr=False)
        else:
            self.lin_edge = None

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = paddle.nn.Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias_attr=False,
            )
        else:
            self.res = None

        if bias:
            self.bias = self.create_parameter([total_out_channels])
        else:
            self.bias = None

        self.reset_parameters()
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.weight.set_value(paddle.nn.initializer.XavierUniform()(self.lin_l.weight.shape))
        self.lin_r.weight.set_value(paddle.nn.initializer.XavierUniform()(self.lin_r.weight.shape))
        if self.lin_edge is not None:
            self.lin_edge.weight.set_value(paddle.nn.initializer.XavierUniform()(self.lin_edge.weight.shape))
        if self.res is not None:
            self.res.weight.set_value(paddle.nn.initializer.XavierUniform()(self.res.weight.shape))
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: Union[Tensor, SparseTensor],
        edge_attr: Optional[Tensor] = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (paddle.Tensor or (paddle.Tensor, paddle.Tensor)): The input node
                features.
            edge_index (paddle.Tensor or SparseTensor): The edge indices.
            edge_attr (paddle.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: Optional[Tensor] = None
        x_r: Optional[Tensor] = None
        if isinstance(x, Tensor):
            assert x.ndim == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).reshape([-1, H, C])
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).reshape([-1, H, C])
        else:
            x_l, x_r = x
            assert x_l.ndim == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).reshape([-1, H, C])
            if x_r is not None:
                x_r = self.lin_r(x_r).reshape([-1, H, C])

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.shape[0]
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.shape[0])
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = edge_index.set_diag()
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.reshape([-1, self.heads * self.out_channels])
        else:
            out = out.mean(axis=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha)
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Optional[Tensor],
                    index: Tensor, ptr: Optional[Tensor],
                    dim_size: Optional[int]) -> Tensor:
        """
        Update edge features.

        Args:
            x_j (Tensor): Source node features.
            x_i (Tensor): Target node features.
            edge_attr (Optional[Tensor]): Edge features.
            index (Tensor): Edge indices.
            ptr (Optional[Tensor]): Pointer tensor for segment operation.
            dim_size (Optional[int]): Dimension size for segment operation.

        Returns:
            Tensor: Updated edge attention scores.
        """
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.reshape([-1, self.heads, self.out_channels])
            x = x + edge_attr

        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        alpha = paddle.sum(x * self.att, axis=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        """
        Compute the message for aggregation.

        Args:
            x_j (Tensor): Source node features.
            alpha (Tensor): Attention scores.

        Returns:
            Tensor: Weighted message for aggregation.
        """
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns:
            str: Class representation.
        """
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

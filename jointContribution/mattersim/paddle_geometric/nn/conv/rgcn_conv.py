from typing import Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.nn import Layer

import paddle_geometric.backend
import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.index import index2ptr
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import glorot, zeros
from paddle_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    paddle_sparse,
)
from paddle_geometric.utils import index_sort, one_hot, scatter, spmm


def masked_edge_index(edge_index: Adj, edge_mask: paddle.Tensor) -> Adj:
    if isinstance(edge_index, paddle.Tensor):
        return edge_index[:, edge_mask]
    return paddle_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGCNConv(nn.Layer):
    r"""
    The relational graph convolutional operator.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        num_blocks (int, optional): If set, this layer will use the
            block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by :obj:`edge_type`. This avoids
            internal re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'mean',
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Cannot apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = self.create_parameter(
                shape=[num_bases, in_channels[0], out_channels],
                default_initializer=paddle.nn.initializer.XavierUniform())
            self.comp = self.create_parameter(
                shape=[num_relations, num_bases],
                default_initializer=paddle.nn.initializer.XavierUniform())

        elif num_blocks is not None:
            if in_channels[0] % num_blocks != 0 or out_channels % num_blocks != 0:
                raise ValueError("Input and output channels must be divisible by num_blocks.")
            self.weight = self.create_parameter(
                shape=[num_relations, num_blocks,
                       in_channels[0] // num_blocks, out_channels // num_blocks],
                default_initializer=paddle.nn.initializer.XavierUniform())
            self.comp = None

        else:
            self.weight = self.create_parameter(
                shape=[num_relations, in_channels[0], out_channels],
                default_initializer=paddle.nn.initializer.XavierUniform())
            self.comp = None

        if root_weight:
            self.root = self.create_parameter(
                shape=[in_channels[1], out_channels],
                default_initializer=paddle.nn.initializer.XavierUniform())
        else:
            self.root = None

        if bias:
            self.bias = self.create_parameter(
                shape=[out_channels],
                default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.bias = None

        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets all learnable parameters of the layer.
        """
        paddle.nn.initializer.XavierUniform()(self.weight)
        if self.comp is not None:
            paddle.nn.initializer.XavierUniform()(self.comp)
        if self.root is not None:
            paddle.nn.initializer.XavierUniform()(self.root)
        if self.bias is not None:
            paddle.nn.initializer.Constant(0.0)(self.bias)
    def forward(
        self,
        x: Union[Optional[paddle.Tensor], Tuple[Optional[paddle.Tensor], paddle.Tensor]],
        edge_index: Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]],
        edge_type: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Runs the forward pass of the module.

        Args:
            x (paddle.Tensor or tuple, optional): The input node features.
                Can be either a `[num_nodes, in_channels]` node feature
                matrix, or an optional one-dimensional node index paddle.Tensor (in
                which case input features are treated as trainable node
                embeddings). Furthermore, `x` can be of type `tuple` denoting
                source and destination node features.
            edge_index (paddle.Tensor or tuple): The edge indices.
            edge_type (paddle.Tensor, optional): The one-dimensional relation type/index
                for each edge in `edge_index`. Should only be `None` in case
                `edge_index` is a Sparsepaddle.Tensor. (default: `None`)
        """
        # Convert input features to a pair of node features or node indices.
        x_l: Optional[paddle.Tensor] = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = paddle.arange(self.in_channels_l, dtype='int64')

        x_r: paddle.Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.shape[0], x_r.shape[0])
        if isinstance(edge_index, paddle.Tensor):
            if edge_type is None:
                raise ValueError("edge_type must be provided when edge_index is a paddle.Tensor.")

        out = paddle.zeros([x_r.shape[0], self.out_channels], dtype=x_r.dtype)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = paddle.matmul(self.comp, weight.reshape([self.num_bases, -1]))
            weight = weight.reshape([self.num_relations, self.in_channels_l, self.out_channels])

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====
            if not x_r.dtype.is_floating_point:
                raise ValueError("Block-diagonal decomposition not supported for non-floating-point features.")

            for i in range(self.num_relations):
                tmp = edge_index[:, edge_type == i]
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.reshape([-1, weight.shape[1], weight.shape[2]])
                h = paddle.matmul(h, weight[i])
                out += h.reshape([-1, self.out_channels])

        else:  # No regularization/Basis-decomposition ========================
            for i in range(self.num_relations):
                tmp = edge_index[:, edge_type == i]

                if not x_r.dtype.is_floating_point:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size)
                else:
                    h = self.propagate(tmp, x=x_l, size=size)
                    out += paddle.matmul(h, weight[i])

        if self.root is not None:
            if not x_r.dtype.is_floating_point:
                out += self.root[x_r]
            else:
                out += paddle.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: paddle.Tensor, edge_type_ptr: Optional[paddle.Tensor]) -> paddle.Tensor:
        if edge_type_ptr is not None:
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return paddle.geometric.segment_matmul(x_j, edge_type_ptr, self.weight)
        return x_j

    def message_and_aggregate(self, adj_t: paddle.Tensor, x: paddle.Tensor) -> paddle.Tensor:
        if isinstance(adj_t, paddle.sparse.Sparsepaddle.Tensor):
            adj_t = adj_t.set_value(None)
        return paddle.geometric.sparse.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
class FastRGCNConv(Layer):
    r"""See :class:`RGCNConv`."""
    def __init__(self, **kwargs):
        super(FastRGCNConv, self).__init__()
        # Initialize parent class and required attributes
        self.aggr = kwargs.get("aggr", "mean")
        self.num_relations = kwargs.get("num_relations", 1)
        self.num_bases = kwargs.get("num_bases", None)
        self.num_blocks = kwargs.get("num_blocks", None)
        self.in_channels_l = kwargs.get("in_channels_l", 1)
        self.out_channels = kwargs.get("out_channels", 1)

        # Parameters
        self.weight = self.create_parameter(
            shape=[self.num_relations, self.in_channels_l, self.out_channels],
            is_bias=False,
        )
        self.comp = self.create_parameter(
            shape=[self.num_bases, self.in_channels_l, self.out_channels],
            is_bias=False,
        ) if self.num_bases else None

        self.root = self.create_parameter(
            shape=[self.in_channels_l, self.out_channels],
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[self.out_channels],
            is_bias=True,
        )

    def forward(self, x: Union[paddle.Tensor, Tuple[Optional[paddle.Tensor], paddle.Tensor]],
                edge_index: paddle.Tensor, edge_type: Optional[paddle.Tensor] = None) -> paddle.Tensor:

        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

        # Convert input features to a pair of node features or node indices.
        x_l = x[0] if isinstance(x, tuple) else x
        if x_l is None:
            x_l = paddle.arange(self.in_channels_l)

        x_r = x_l if not isinstance(x, tuple) else x[1]
        size = (x_l.shape[0], x_r.shape[0])

        # propagate_type: (x: paddle.Tensor, edge_type: paddle.Tensor)
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size)

        if self.root is not None:
            if x_r.dtype != paddle.float32:
                out = out + self.root[x_r]
            else:
                out = out + paddle.matmul(x_r, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: paddle.Tensor, edge_type: paddle.Tensor,
                edge_index_j: paddle.Tensor) -> paddle.Tensor:
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = paddle.matmul(self.comp, weight.reshape([self.num_bases, -1]))
            weight = weight.reshape(
                [self.num_relations, self.in_channels_l, self.out_channels])

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if x_j.dtype != paddle.float32:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            weight = weight[edge_type].reshape(
                [-1, weight.shape[1], weight.shape[2]])
            x_j = x_j.reshape([-1, 1, weight.shape[1]])
            return paddle.matmul(x_j, weight).reshape([-1, self.out_channels])

        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype != paddle.float32:
                weight_index = edge_type * weight.shape[1] + edge_index_j
                return weight.reshape([-1, self.out_channels])[weight_index]

            return paddle.matmul(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, inputs: paddle.Tensor, edge_type: paddle.Tensor, index: paddle.Tensor,
                  dim_size: Optional[int] = None) -> paddle.Tensor:
        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            norm = one_hot(edge_type, self.num_relations, dtype=inputs.dtype)
            norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            norm = paddle.gather(norm, edge_type.reshape([-1, 1]), axis=1)
            norm = 1. / paddle.clip(norm, min=1.)
            inputs = norm * inputs

        return scatter(inputs, index, axis=self.node_dim, num=dim_size)
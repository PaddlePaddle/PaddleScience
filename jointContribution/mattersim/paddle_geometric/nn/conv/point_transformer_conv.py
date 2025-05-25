from typing import Callable, Optional, Tuple, Union

import paddle
from paddle import Tensor
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.dense import Linear
from paddle_geometric.nn.inits import reset
from paddle_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    paddle_sparse,
)
from paddle_geometric.utils import add_self_loops, remove_self_loops, softmax


class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        pos_nn (paddle.nn.Layer, optional): A neural network
            which maps relative spatial coordinates to the output shape.
        attn_nn (paddle.nn.Layer, optional): A neural network that maps
            transformed node features to the output shape.
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`paddle_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn if pos_nn is not None else Linear(3, out_channels)
        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels, bias_attr=False)
        self.lin_src = Linear(in_channels[0], out_channels, bias_attr=False)
        self.lin_dst = Linear(in_channels[1], out_channels, bias_attr=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:

        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].shape[0], pos[1].shape[0]))
            elif isinstance(edge_index, SparseTensor):
                edge_index = paddle_sparse.set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        delta = self.pos_nn(pos_i - pos_j)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha * (x_j + delta)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

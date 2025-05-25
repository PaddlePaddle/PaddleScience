from typing import Callable, Optional, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import reset
from paddle_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor
from paddle_geometric.utils import add_self_loops, remove_self_loops


class PointNetConv(MessagePassing):
    r"""The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
    for 3D Classification and Segmentation"
    <https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ papers.

    Args:
        local_nn (Callable, optional): A neural network that maps node features
            and relative spatial coordinates of shape `[-1, in_channels + num_dimensions]`
            to shape `[-1, out_channels]`.
        global_nn (Callable, optional): A neural network that maps aggregated node
            features of shape `[-1, out_channels]` to shape `[-1, final_out_channels]`.
        add_self_loops (bool, optional): If set to `False`, will not add self-loops to the input graph.
            (default: `True`)
        **kwargs (optional): Additional arguments of `paddle_geometric.nn.conv.MessagePassing`.

    """
    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(
        self,
        x: Union[OptTensor, PairOptTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:

        if not isinstance(x, tuple):
            x = (x, None)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].shape[0], pos[1].shape[0]))
            elif isinstance(edge_index, SparseTensor):
                edge_index = edge_index.set_diag()

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if x_j is not None:
            msg = paddle.concat([x_j, msg], axis=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')

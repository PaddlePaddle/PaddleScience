from typing import Callable, Optional, Union

import paddle
from paddle import Tensor
from paddle_geometric.nn.conv import MessagePassing
from paddle_geometric.nn.inits import reset
from paddle_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor, SparseTensor
from paddle_geometric.utils import add_self_loops, remove_self_loops


def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return paddle.atan2(
        paddle.norm(paddle.cross(v1, v2, axis=1), p=2, axis=1),
        paddle.sum(v1 * v2, axis=1)
    )


def point_pair_features(pos_i: Tensor, pos_j: Tensor, normal_i: Tensor,
                        normal_j: Tensor) -> Tensor:
    pseudo = pos_j - pos_i
    return paddle.stack([
        paddle.norm(pseudo, p=2, axis=1),
        get_angle(normal_i, pseudo),
        get_angle(normal_j, pseudo),
        get_angle(normal_i, normal_j)
    ], axis=1)


class PPFConv(MessagePassing):
    r"""The PPFNet operator from the `"PPFNet: Global Context Aware Local
    Features for Robust 3D Point Matching" <https://arxiv.org/abs/1802.02669>`_
    paper.
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
        normal: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:

        if not isinstance(x, tuple):
            x = (x, None)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if isinstance(normal, Tensor):
            normal = (normal, normal)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].shape[0])
            elif isinstance(edge_index, SparseTensor):
                edge_index = paddle_sparse.set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor, normal: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, normal=normal)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: OptTensor, pos_i: Tensor, pos_j: Tensor,
                normal_i: Tensor, normal_j: Tensor) -> Tensor:
        msg = point_pair_features(pos_i, pos_j, normal_i, normal_j)
        if x_j is not None:
            msg = paddle.concat([x_j, msg], axis=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')

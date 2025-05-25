from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Linear

class DenseSAGEConv(paddle.nn.Layer):
    r"""See :class:`paddle_geometric.nn.conv.SAGEConv`.

    .. note::

        :class:`~paddle_geometric.nn.dense.DenseSAGEConv` expects to work on
        binary adjacency matrices.
        If you want to make use of weighted dense adjacency matrices, please
        use :class:`paddle_geometric.nn.dense.DenseGraphConv` instead.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, out_channels, bias_attr=False)
        self.lin_root = Linear(in_channels, out_channels, bias_attr=bias)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for layer in self.sublayers():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.ndim == 2 else x
        adj = adj.unsqueeze(0) if adj.ndim == 2 else adj
        B, N, _ = adj.shape

        out = paddle.matmul(adj, x)
        out = out / paddle.clip(adj.sum(axis=-1, keepdim=True), min=1)
        out = self.lin_rel(out) + self.lin_root(x)

        if self.normalize:
            out = F.normalize(out, p=2.0, axis=-1)

        if mask is not None:
            out = out * mask.reshape([B, N, 1]).astype(x.dtype)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

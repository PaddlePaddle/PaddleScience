from typing import Optional

import paddle
from paddle import Tensor
from paddle.nn import Layer

class DenseGINConv(Layer):
    r"""See :class:`paddle_geometric.nn.conv.GINConv`."""
    def __init__(
        self,
        nn: Layer,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(eps))
        else:
            self.eps = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(eps), is_bias=False, stop_gradient=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for layer in self.sublayers():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if not self.eps.stop_gradient:
            self.eps.set_value(paddle.full(shape=[1], fill_value=self.initial_eps))

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None,
                add_loop: bool = True) -> Tensor:
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
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.ndim == 2 else x
        adj = adj.unsqueeze(0) if adj.ndim == 2 else adj
        B, N, _ = adj.shape

        out = paddle.matmul(adj, x)
        if add_loop:
            out = (1 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * mask.reshape([B, N, 1]).astype(x.dtype)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

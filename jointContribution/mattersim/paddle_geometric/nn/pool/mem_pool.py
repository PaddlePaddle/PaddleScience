import warnings
from typing import Optional, Tuple

import paddle
from paddle import Tensor
from paddle.nn import Conv2D, KLDivLoss, Linear

from paddle_geometric.utils import to_dense_batch

EPS = 1e-15


class MemPooling(paddle.nn.Layer):
    r"""Memory based pooling layer from `"Memory-Based Graph Networks"
    <https://arxiv.org/abs/2002.09518>`_ paper, which learns a coarsened graph
    representation based on soft cluster assignments.

    .. math::
        S_{i,j}^{(h)} &= \frac{
        (1+{\| \mathbf{x}_i-\mathbf{k}^{(h)}_j \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}{
        \sum_{k=1}^K (1 + {\| \mathbf{x}_i-\mathbf{k}^{(h)}_k \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}

        \mathbf{S} &= \textrm{softmax}(\textrm{Conv2d}
        (\Vert_{h=1}^H \mathbf{S}^{(h)})) \in \mathbb{R}^{N \times K}

        \mathbf{X}^{\prime} &= \mathbf{S}^{\top} \mathbf{X} \mathbf{W} \in
        \mathbb{R}^{K \times F^{\prime}}

    where :math:`H` denotes the number of heads, and :math:`K` denotes the
    number of clusters.

    Args:
        in_channels (int): Size of each input sample :math:`F`.
        out_channels (int): Size of each output sample :math:`F^{\prime}`.
        heads (int): The number of heads :math:`H`.
        num_clusters (int): number of clusters :math:`K` per head.
        tau (int, optional): The temperature :math:`\tau`. (default: :obj:`1.`)
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 num_clusters: int, tau: float = 1.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_clusters = num_clusters
        self.tau = tau

        self.k = paddle.create_parameter(shape=[heads, num_clusters, in_channels], dtype='float32')
        self.conv = Conv2D(heads, 1, kernel_size=1, padding=0, bias_attr=False)
        self.lin = Linear(in_channels, out_channels, bias_attr=False)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        paddle.nn.initializer.uniform_(self.k, -1., 1.)
        self.conv.reset_parameters()
        self.lin.reset_parameters()

    @staticmethod
    def kl_loss(S: Tensor) -> Tensor:
        r"""The additional KL divergence-based loss.

        .. math::
            P_{i,j} &= \frac{S_{i,j}^2 / \sum_{n=1}^N S_{n,j}}{\sum_{k=1}^K
            S_{i,k}^2 / \sum_{n=1}^N S_{n,k}}

            \mathcal{L}_{\textrm{KL}} &= \textrm{KLDiv}(\mathbf{P} \Vert
            \mathbf{S})
        """
        S_2 = S**2
        P = S_2 / S.sum(axis=1, keepdim=True)
        denom = P.sum(axis=2, keepdim=True)
        denom[S.sum(axis=2, keepdim=True) == 0.0] = 1.0
        P /= denom

        loss = KLDivLoss(reduction='batchmean', log_target=False)
        return loss(paddle.clamp(S, EPS).log(), paddle.clamp(P, EPS))

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        max_num_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The node feature tensor of shape
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}` or
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`.
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
                Should not be provided in case node features already have shape
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`.
                (default: :obj:`None`)
            mask (paddle.Tensor, optional): A mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}`, which
                indicates valid nodes for each graph when using
                node features of shape
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`.
                (default: :obj:`None`)
            max_num_nodes (int, optional): The size of the :math:`B` node
                dimension. Automatically calculated if not given.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        if x.dim() <= 2:
            x, mask = to_dense_batch(x, batch, max_num_nodes=max_num_nodes,
                                     batch_size=batch_size)
        elif mask is None:
            mask = paddle.ones((x.shape[0], x.shape[1]), dtype=paddle.bool)

        (B, N, _), H, K = x.shape, self.heads, self.num_clusters

        dist = paddle.cdist(self.k.reshape([H * K, -1]), x.reshape([B * N, -1]), p=2)**2
        dist = (1. + dist / self.tau).pow(-(self.tau + 1.0) / 2.0)

        dist = dist.reshape([H, K, B, N]).transpose([2, 0, 3, 1])  # [B, H, N, K]
        S = dist / dist.sum(axis=-1, keepdim=True)

        S = self.conv(S).squeeze(axis=1).softmax(axis=-1)  # [B, N, K]
        S = S * mask.reshape([B, N, 1])

        x = self.lin(S.transpose([0, 2, 1]) @ x)

        return x, S

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'num_clusters={self.num_clusters})')

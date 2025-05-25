from typing import Optional, Tuple

import paddle
from paddle import Tensor


def dense_diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper.

    Args:
        x (Tensor): Node feature tensor of shape
            :math:`[B, N, F]`, where `B` is the batch size,
            `N` is the number of nodes, and `F` is the feature dimension.
        adj (Tensor): Adjacency tensor of shape `[B, N, N]`.
        s (Tensor): Assignment tensor of shape `[B, N, C]`
            where `C` is the number of clusters.
        mask (Tensor, optional): Mask tensor of shape `[B, N]` indicating
            the valid nodes for each graph.
        normalize (bool, optional): If `False`, the link prediction loss is
            not divided by `adj.numel()`. Defaults to `True`.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Pooled node feature matrix,
        coarsened adjacency matrix, link prediction loss, and entropy regularization.
    """
    x = x.unsqueeze(0) if x.ndim == 2 else x
    adj = adj.unsqueeze(0) if adj.ndim == 2 else adj
    s = s.unsqueeze(0) if s.ndim == 2 else s

    batch_size, num_nodes, _ = x.shape

    s = paddle.nn.functional.softmax(s, axis=-1)

    if mask is not None:
        mask = mask.reshape([batch_size, num_nodes, 1]).astype(x.dtype)
        x, s = x * mask, s * mask

    out = paddle.matmul(s.transpose([0, 2, 1]), x)
    out_adj = paddle.matmul(paddle.matmul(s.transpose([0, 2, 1]), adj), s)

    link_loss = adj - paddle.matmul(s, s.transpose([0, 2, 1]))
    link_loss = paddle.norm(link_loss, p=2)
    if normalize:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * paddle.log(s + 1e-15)).sum(axis=-1).mean()

    return out, out_adj, link_loss, ent_loss

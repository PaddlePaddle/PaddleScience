from typing import Optional, Tuple
import paddle
from paddle import Tensor

def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and two auxiliary objectives: (1) The MinCut
    loss

    .. math::
        \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F.

    Args:
        x (Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied beforehand, since it is
            executed within this method.
        mask (Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        temp (float, optional): Temperature parameter for softmax function.
            (default: :obj:`1.0`)

    :rtype: (:class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`, :class:`Tensor`)
    """
    x = x.unsqueeze(0) if x.ndim == 2 else x
    adj = adj.unsqueeze(0) if adj.ndim == 2 else adj
    s = s.unsqueeze(0) if s.ndim == 2 else s

    (batch_size, num_nodes, _), k = x.shape, s.shape[-1]

    s = paddle.nn.functional.softmax(s / temp if temp != 1.0 else s, axis=-1)

    if mask is not None:
        mask = mask.reshape([batch_size, num_nodes, 1]).astype(x.dtype)
        x, s = x * mask, s * mask

    out = paddle.matmul(s.transpose([0, 2, 1]), x)
    out_adj = paddle.matmul(paddle.matmul(s.transpose([0, 2, 1]), adj), s)

    # MinCut regularization
    mincut_num = _rank3_trace(out_adj)
    d_flat = paddle.sum(adj, axis=-1)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(paddle.matmul(paddle.matmul(s.transpose([0, 2, 1]), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = paddle.mean(mincut_loss)

    # Orthogonality regularization
    ss = paddle.matmul(s.transpose([0, 2, 1]), s)
    i_s = paddle.eye(k, dtype=ss.dtype)
    ortho_loss = paddle.norm(
        ss / paddle.norm(ss, axis=[-1, -2], keepdim=True) -
        i_s / paddle.norm(i_s), axis=[-1, -2]
    )
    ortho_loss = paddle.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix
    ind = paddle.arange(k)
    out_adj[:, ind, ind] = 0
    d = paddle.sum(out_adj, axis=-1)
    d = paddle.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose([0, 2, 1])

    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return paddle.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = paddle.eye(x.shape[1], dtype=x.dtype)
    out = eye * x.unsqueeze(2).expand([x.shape[0], x.shape[1], x.shape[1]])
    return out

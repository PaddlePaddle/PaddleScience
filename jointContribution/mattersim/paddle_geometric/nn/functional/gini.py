import paddle

def gini(w: paddle.Tensor) -> paddle.Tensor:
    r"""The Gini coefficient from the `"Improving Molecular Graph Neural
    Network Explainability with Orthonormalization and Induced Sparsity"
    <https://arxiv.org/abs/2105.04854>`_ paper.

    Computes a regularization penalty :math:`\in [0, 1]` for each row of a
    matrix according to

    .. math::
        \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
         - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

    and returns an average over all rows.

    Args:
        w (paddle.Tensor): A two-dimensional tensor.
    """
    s = 0
    for row in w:
        t = row.expand([row.shape[0], row.shape[0]])
        u = (paddle.abs(t - t.transpose([1, 0])).sum() /
             (2 * (row.shape[0]**2 - row.shape[0]) *
              paddle.mean(paddle.abs(row)) + paddle.finfo(row.dtype).eps))
        s += u
    s /= w.shape[0]
    return s

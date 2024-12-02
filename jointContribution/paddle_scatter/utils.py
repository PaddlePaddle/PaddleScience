import paddle


def broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int) -> paddle.Tensor:
    r"""Broadcast `src` to `other` at dimension `dim`.

    Denote dim = :math:`i`,
    other.shape = :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`,
    src.shape = :math:(x_i,)`, src = :math:`[y_0, ..., y_{x_i-1}]`,
    where each element satisfying 0 <= element < x_i

    This util function broadcast `src` to the shape of `other`'s.

    Notes:
        The shape of `src` should be either :math:`(x_i,)` or :math:`(*_0, ..., *_{i-1}, x_i)`,
        where :math:`*_k (k <= i-1)` should be either :math:`1` or :math:`x_k`.

    Args:
        src (paddle.Tensor): The tensor to be broadcasted.
        other (paddle.Tensor): The tensor to be broadcasted to.
        dim (int): The target dimension of `other`.

    Returns:
        paddle.Tensor, the broadcasted tensor.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    if other.numel() == 0:
        return src.reshape(other.shape)
    src = src.expand(other.shape)
    return src

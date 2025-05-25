import paddle
from paddle import Tensor

def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`paddle.cumsum`, prepends the output with zero.

    Args:
        x (paddle.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = paddle.to_tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    # Create a tensor with an additional element in the specified dimension
    size = tuple(x.shape[:dim]) + (x.shape[dim] + 1,) + tuple(x.shape[dim + 1:])
    out = paddle.zeros(size, dtype=x.dtype)

    # Compute the cumulative sum, excluding the first element (zero)
    if x.shape[dim] > 0:
        out.slice([dim], [1], [x.shape[dim] + 1])[:] = paddle.cumsum(x, axis=dim)

    return out

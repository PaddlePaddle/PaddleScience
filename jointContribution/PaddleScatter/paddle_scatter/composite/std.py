from typing import Optional

import paddle
from paddle_scatter.scatter import scatter_sum
from paddle_scatter.utils import broadcast


def scatter_std(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is std. (If dtype of `src` is int, output is still int.)

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The indices of elements to scatter. The dimension
            of index should either be 1-D or :math:`i+1`-D. See Notes for more
            details.
        dim (int, optional): The axis along which to index. Default is -1.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.
        unbiased (bool, optional): Indicate whether to calculate biased std (divide by n)
            or unbiased std (divide by n-1). Default is True.

    Returns:
        paddle.Tensor, the reduced tensor by std reduction method.
    """
    if out is not None:
        dim_size = out.shape[dim]

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = paddle.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = broadcast(count, tmp, dim).clip(1)
    mean = tmp.divide(count)

    var = src - mean.take_along_axis(indices=index, axis=dim)
    var = var * var
    res = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count = count.subtract(paddle.full([], 1, dtype=src.dtype)).clip(1)
    res = res.divide(count + 1e-6).sqrt()

    if out is not None:
        paddle.assign(res, out)
        return out
    else:
        return res

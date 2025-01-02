from typing import Optional

import paddle
from paddle_scatter.scatter import scatter_max
from paddle_scatter.scatter import scatter_sum
from paddle_scatter.utils import broadcast


def scatter_logsumexp(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    eps: float = 1e-12,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is logsumexp. (If dtype of `src` is int, output is still int.)

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
        eps (float, optional): Eplison factor added to the sum of exponent values during
            computation in case they are zero. Default is 1e-12.

    Returns:
        paddle.Tensor, the reduced tensor by logsumexp reduction method.
    """
    if not paddle.is_floating_point(src):
        raise ValueError(
            "`scatter_logsumexp` can only be computed over "
            "tensors with floating point data types."
        )

    index = broadcast(index, src, dim)
    eps = paddle.full([], eps, dtype=src.dtype)

    if out is not None:
        dim_size = out.shape[dim]
    else:
        if dim_size is None:
            dim_size = int(index.max()) + 1

    size = src.shape
    size[dim] = dim_size
    max_value_per_index = paddle.full(
        size,
        fill_value=float("-inf"),
        dtype=src.dtype,
    )
    scatter_max(src, index, dim, max_value_per_index, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.take_along_axis(indices=index, axis=dim)
    recentered_score = src - max_per_src_element
    recentered_score.masked_fill_(paddle.isnan(recentered_score), float("-inf"))

    orig_out: Optional[paddle.Tensor] = None
    if out is not None:
        orig_out = out.clone()
        res = out.subtract(max_value_per_index).exp()

        sum_per_index = scatter_sum(recentered_score.exp(), index, dim, res, dim_size)
    else:
        sum_per_index = scatter_sum(recentered_score.exp(), index, dim, None, dim_size)

    res = sum_per_index.add(eps).log().add(max_value_per_index)

    if orig_out is None:
        return res.nan_to_num_(neginf=0.0)

    mask = ~res.isfinite()
    res[mask] = orig_out[mask]
    paddle.assign(res, out)
    return out

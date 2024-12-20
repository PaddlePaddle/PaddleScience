from typing import Optional

import paddle
from paddle_scatter.scatter import scatter_max
from paddle_scatter.scatter import scatter_sum
from paddle_scatter.utils import broadcast


def scatter_softmax(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is softmax. (If dtype of `src` is int, output is still int.)

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The indices of elements to scatter. The dimension
            of index should either be 1-D or :math:`i+1`-D. See Notes for more
            details.
        dim (int, optional): The axis along which to index. Default is -1.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by softmax reduction method.
    """
    if not paddle.is_floating_point(src):
        raise ValueError(
            "`scatter_softmax` can only be computed over tensors "
            "with floating point data types."
        )

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.take_along_axis(indices=index, axis=dim)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = scatter_sum(recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.take_along_axis(indices=index, axis=dim)

    return recentered_scores_exp.divide(normalizing_constants)


def scatter_log_softmax(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is log_softmax. (If dtype of `src` is int, output is still int.)

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The indices of elements to scatter. The dimension
            of index should either be 1-D or :math:`i+1`-D. See Notes for more
            details.
        dim (int, optional): The axis along which to index. Default is -1.
        eps (float, optional): Eplison factor added to the normalizing constants during
            computation in case they are zero. Default is 1e-12.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by log_softmax reduction method.
    """
    if not paddle.is_floating_point(src):
        raise ValueError(
            "`scatter_log_softmax` can only be computed over "
            "tensors with floating point data types."
        )

    index = broadcast(index, src, dim)
    eps = paddle.full([], eps, dtype=src.dtype)

    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.take_along_axis(indices=index, axis=dim)

    recentered_scores = src - max_per_src_element

    sum_per_index = scatter_sum(recentered_scores.exp(), index, dim, dim_size=dim_size)
    normalizing_constants = (
        sum_per_index.add(eps).log().take_along_axis(indices=index, axis=dim)
    )

    return recentered_scores.subtract(normalizing_constants)

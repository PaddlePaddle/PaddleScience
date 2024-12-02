from typing import Optional
from typing import Tuple

import paddle
import paddle_scatter_ops

from .utils import broadcast


def scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is sum.

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

    Returns:
        paddle.Tensor, the reduced tensor by sum reduction method.
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = src.shape
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        arr = paddle.zeros(size, dtype=src.dtype)
        if src.numel() == 0:
            return arr
        return paddle.put_along_axis(
            arr, indices=index, values=src, axis=dim, reduce="add"
        )
    else:
        if src.numel() == 0:
            return out
        result = paddle.put_along_axis(
            out, indices=index, values=src, axis=dim, reduce="add"
        )
        paddle.assign(result, out)
        return out


def scatter_add(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is sum.

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

    Returns:
        paddle.Tensor, the reduced tensor by sum reduction method.
    """
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is multiply.

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

    Returns:
        paddle.Tensor, the reduced tensor by multiply reduction method.
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = src.shape
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        arr = paddle.ones(size, dtype=src.dtype)
        if src.numel() == 0:
            return arr
        return paddle.put_along_axis(
            arr, indices=index, values=src, axis=dim, reduce="mul"
        )
    else:
        if src.numel() == 0:
            return out
        result = paddle.put_along_axis(
            out, indices=index, values=src, axis=dim, reduce="mul"
        )
        paddle.assign(result, out)
        return out


def scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is mean. (If dtype of `src` is int, output is still int.)

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

    Returns:
        paddle.Tensor, the reduced tensor by mean reduction method.
    """
    sums = scatter_sum(src, index, dim, out, dim_size)
    dim_size = sums.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones_tensor = paddle.ones(index.shape, dtype=src.dtype)
    tmp = scatter_sum(ones_tensor, index, index_dim, None, dim_size)
    count = paddle.where(tmp < 1, paddle.full_like(tmp, 1), tmp, name="where")
    count = broadcast(count, sums, dim)
    if sums.is_floating_point():
        result = paddle.divide(sums, count)
    else:
        result = paddle.floor_divide(sums, count)
    if out is None:
        return result
    else:
        paddle.assign(result, out)
        return out


def scatter_min(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is min.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The indices of elements to scatter. The dimension
            of index should either be 1-D or :math:`i+1`-D. See Notes for more
            details.
        dim (int, optional): The axis along which to index. Default is -1.
        out (paddle.Tensor|None, optional): The destination tensor of min result.
            Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor], the reduced min tensor and arg_min tensor.
    """
    if dim < 0:
        dim = dim + index.dim()
    index = broadcast(index, src, dim)
    size = src.shape
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1

    if out is None:
        if src.numel() == 0:
            return (
                paddle.zeros(size, dtype=src.dtype),
                paddle.full(size, src.shape[dim], index.dtype),
            )
        return paddle_scatter_ops.custom_scatter_min_max(
            src, index, None, size, "min", dim
        )
    else:
        if src.numel() == 0:
            return (out, paddle.full(size, src.shape[dim], index.dtype))
        for i in range(len(size)):
            if i != dim:
                assert size[i] == out.shape[i]
        result, arg_result = paddle_scatter_ops.custom_scatter_min_max(
            src, index, out, out.shape, "min", dim
        )
        paddle.assign(result, out)
        return out, arg_result


def scatter_max(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`,
    the reduction method is max.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The indices of elements to scatter. The dimension
            of index should either be 1-D or :math:`i+1`-D. See Notes for more
            details.
        dim (int, optional): The axis along which to index. Default is -1.
        out (paddle.Tensor|None, optional): The destination tensor of max result.
            Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor], the reduced max tensor and arg_max tensor.
    """
    if dim < 0:
        dim = dim + index.dim()
    index = broadcast(index, src, dim)
    size = src.shape
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1

    if out is None:
        if src.numel() == 0:
            return (
                paddle.zeros(size, dtype=src.dtype),
                paddle.full(size, src.shape[dim], index.dtype),
            )
        return paddle_scatter_ops.custom_scatter_min_max(
            src, index, None, size, "max", dim
        )
    else:
        if src.numel() == 0:
            return (out, paddle.full(size, src.shape[dim], index.dtype))
        for i in range(len(size)):
            if i != dim:
                assert size[i] == out.shape[i]
        result, arg_result = paddle_scatter_ops.custom_scatter_min_max(
            src, index, out, out.shape, "max", dim
        )
        paddle.assign(result, out)
        return out, arg_result


def scatter(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along a given axis`dim`.

    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `dim` and by the corresponding
    value in `index` for dimension `dim`. The applied reduction is defined
    via the `reduce` argument.

    Formally, if `src` and `index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and `dim` = `i`, then `out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of `index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The `index` tensor supports broadcasting in case its dimensions do
    not match with `src`.

    For one-dimensional tensors with `reduce="sum"`, the operation
    computes

    $$
    \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j
    $$

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Notes:
        `index` tensor supports broadcasting, and its shape should be either
        :math:`(x_i, )` or :math:`(*_0, ..., *_{i-1}, x_i)`,
        where :math:`*_k (k <= i-1)` should be either :math:`1` or :math:`x_k`.

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
        reduce (str, optional): The reduce operation supports `"sum"`, `"add"`, `"mul"`,
            `"mean"`, `"min"` or `"max"`. Default is `"sum"`.

    Returns:
        paddle.Tensor, the reduced tensor.

    Examples:
        >>> from paddle_scatter import scatter

        >>> src = paddle.randn([10, 6, 64])
        >>> index = paddle.tensor([0, 1, 0, 1, 2, 1])

        >>> # Broadcasting in the first and last dim
        >>> out = scatter(src, index, dim=1, reduce="sum")
        >>> print(out.shape)
        [10, 3, 64]

        >>> # Specify `dim_size`
        >>> out = scatter(src, index, dim=1, dim_size=4, reduce="sum")
        >>> print(out.shape)
        [10, 4, 64]

        >>> # Specify `out`
        >>> out = paddle.empty([10, 3, 64])
        >>> scatter(src, index, dim=1, out=out, reduce="sum")
        >>> print(out.shape)
        [10, 3, 64]
    """
    if reduce == "sum" or reduce == "add":
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == "mul":
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == "min":
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == "max":
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

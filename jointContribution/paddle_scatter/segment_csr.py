from typing import Optional
from typing import Tuple

import paddle
from paddle import arange
from paddle import assign
from paddle import full
from paddle import repeat_interleave
from paddle import zeros
from paddle.geometric import segment_mean
from paddle.geometric import segment_sum
from paddle_scatter_min_max_ops import custom_segment_csr_min_max

from .utils import transform_2d
from .utils import transform_3d


def segment_sum_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""
    Reduces all values from the `src` tensor into `out` within the
    ranges specified in the `indptr` tensor along the last dimension of
    `indptr`.
    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `indptr.dim() - 1` and by the
    corresponding range index in `indptr` for dimension
    `indptr.dim() - 1`.
    The reduction method is sum.

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by sum reduction method.
    """
    indptr_shape = indptr.shape
    src_shape = src.shape
    dim = len(indptr_shape) - 1
    # broadcast indptr to src
    indptr_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        indptr = indptr.reshape(indptr_shape)
    else:
        indptr = indptr.expand(indptr_shape)

    num_seg = indptr_shape[dim] - 1
    if out is None:
        out_size = src_shape
        if indptr.numel() == 0:
            out_size[dim] = 0
        else:
            out_size[dim] = num_seg
        if src.numel() == 0:
            return zeros(out_size, dtype=src.dtype)
    else:
        assert (
            out.shape[dim] == num_seg
        ), "The (size of indptr at last dimension) must be\
                                             equal to the (size of out at the same dimension) + 1."
        if src.numel() == 0:
            return out
        out_size = out.shape
    tmp = zeros(out_size, dtype=src.dtype)

    repeats = indptr.diff(n=1, axis=dim)
    assert (
        repeats.sum(axis=dim) == src.shape[dim]
    ).all(), "The length of specified index by indptr shoud be\
                                                             equal to the size of src at last dimension of indptr."
    src_flatten = transform_3d(src, dim)
    out_flatten = transform_3d(tmp, dim)
    repeats_flatten = transform_2d(repeats, dim)
    src_dim_indices = arange(num_seg)
    for i in range(src_flatten.shape[0]):
        for j in range(src_flatten.shape[-1]):
            belongs_to = repeat_interleave(src_dim_indices, repeats_flatten[i], 0)
            result = segment_sum(src_flatten[i, :, j], belongs_to)
            if out_size[dim] >= len(result):
                out_flatten[i, : len(result), j] = result
            else:
                out_flatten[i, :, j] = result[: out_size[dim]]
    if out is None:
        return out_flatten.reshape(out_size)
    else:
        assign(out_flatten.reshape(out_size), out)
        return out


def segment_add_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""
    Reduces all values from the `src` tensor into `out` within the
    ranges specified in the `indptr` tensor along the last dimension of
    `indptr`.
    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `indptr.dim() - 1` and by the
    corresponding range index in `indptr` for dimension
    `indptr.dim() - 1`.
    The reduction method is sum.

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by sum reduction method.
    """
    return segment_sum_csr(src, indptr, out)


def segment_mean_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""
    Reduces all values from the `src` tensor into `out` within the
    ranges specified in the `indptr` tensor along the last dimension of
    `indptr`.
    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `indptr.dim() - 1` and by the
    corresponding range index in `indptr` for dimension
    `indptr.dim() - 1`.
    The reduction method is mean.

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by mean reduction method.
    """
    indptr_shape = indptr.shape
    src_shape = src.shape
    dim = len(indptr_shape) - 1
    # broadcast indptr to src
    indptr_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        indptr = indptr.reshape(indptr_shape)
    else:
        indptr = indptr.expand(indptr_shape)

    num_seg = indptr_shape[dim] - 1
    if out is None:
        out_size = src_shape
        if indptr.numel() == 0:
            out_size[dim] = 0
        else:
            out_size[dim] = num_seg
        if src.numel() == 0:
            return zeros(out_size, dtype=src.dtype)
    else:
        assert (
            out.shape[dim] == num_seg
        ), "The (size of indptr at last dimension) must be\
                                             equal to the (size of out at the same dimension) + 1."
        if src.numel() == 0:
            return out
        out_size = out.shape
    tmp = zeros(out_size, dtype=src.dtype)

    repeats = indptr.diff(n=1, axis=dim)
    assert (
        repeats.sum(axis=dim) == src.shape[dim]
    ).all(), "The length of specified index by indptr shoud be\
                                                             equal to the size of src at last dimension of indptr."
    src_flatten = transform_3d(src, dim)
    out_flatten = transform_3d(tmp, dim)
    repeats_flatten = transform_2d(repeats, dim)
    src_dim_indices = arange(num_seg)
    for i in range(src_flatten.shape[0]):
        for j in range(src_flatten.shape[-1]):
            belongs_to = repeat_interleave(src_dim_indices, repeats_flatten[i], 0)
            result = segment_mean(src_flatten[i, :, j], belongs_to)
            if out_size[dim] >= len(result):
                out_flatten[i, : len(result), j] = result
            else:
                out_flatten[i, :, j] = result[: out_size[dim]]
    if out is None:
        return out_flatten.reshape(out_size)
    else:
        assign(out_flatten.reshape(out_size), out)
        return out


def segment_min_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""
    Reduces all values from the `src` tensor into `out` within the
    ranges specified in the `indptr` tensor along the last dimension of
    `indptr`.
    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `indptr.dim() - 1` and by the
    corresponding range index in `indptr` for dimension
    `indptr.dim() - 1`.
    The reduction method is min.

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor], the reduced min tensor and arg_min tensor.
    """
    indptr_shape = indptr.shape
    src_shape = src.shape
    dim = len(indptr_shape) - 1
    # broadcast indptr to src
    indptr_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        indptr = indptr.reshape(indptr_shape)
    else:
        indptr = indptr.expand(indptr_shape)

    if out is None:
        size = src.shape
        size[dim] = max(indptr_shape[dim] - 1, 0)
        if src.numel() == 0:
            return (
                zeros(size, dtype=src.dtype),
                full(size, src.shape[dim], indptr.dtype),
            )
        return custom_segment_csr_min_max(src, indptr, size, "min")
    else:
        if src.numel() == 0:
            return (out, full(size, src.shape[dim], indptr.dtype))
        result, arg_result = custom_segment_csr_min_max(src, indptr, out.shape, "min")
        assign(result, out)
        return out, arg_result


def segment_max_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""
    Reduces all values from the `src` tensor into `out` within the
    ranges specified in the `indptr` tensor along the last dimension of
    `indptr`.
    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `indptr.dim() - 1` and by the
    corresponding range index in `indptr` for dimension
    `indptr.dim() - 1`.
    The reduction method is max.

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor], the reduced max tensor and arg_max tensor.
    """
    indptr_shape = indptr.shape
    src_shape = src.shape
    dim = len(indptr_shape) - 1
    # broadcast indptr to src
    indptr_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        indptr = indptr.reshape(indptr_shape)
    else:
        indptr = indptr.expand(indptr_shape)

    if out is None:
        size = src.shape
        size[dim] = max(indptr_shape[dim] - 1, 0)
        if src.numel() == 0:
            return (
                zeros(size, dtype=src.dtype),
                full(size, src.shape[dim], indptr.dtype),
            )
        return custom_segment_csr_min_max(src, indptr, size, "max")
    else:
        if src.numel() == 0:
            return (out, full(size, src.shape[dim], indptr.dtype))
        result, arg_result = custom_segment_csr_min_max(src, indptr, out.shape, "max")
        assign(result, out)
        return out, arg_result


def segment_csr(
    src: paddle.Tensor,
    indptr: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    r"""
    Reduces all values from the `src` tensor into `out` within the
    ranges specified in the `indptr` tensor along the last dimension of
    `indptr`.
    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `indptr.dim() - 1` and by the
    corresponding range index in `indptr` for dimension
    `indptr.dim() - 1`.
    The applied reduction is defined via the `reduce` argument.

    Formally, if `src` and `indptr` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_1, ..., x_{m-1}, x_m, x_{m+1}, ..., x_n)` and
    :math:`(x_1, ..., x_{m-1}, y)`, respectively, then `out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_1, ..., x_{m-1}, y - 1, x_{m+1}, ..., x_n)`.
    Moreover, the values of `indptr` must be between :math:`0` and
    :math:`x_m` in ascending order.
    The `indptr` tensor supports broadcasting in case its dimensions do
    not match with `src`.

    For one-dimensional tensors with `reduce="sum"`, the operation
    computes

    $$
    \mathrm{out}_i =
    \sum_{j = \mathrm{indptr}[i]}^{\mathrm{indptr}[i+1]-1}~\mathrm{src}_j.
    $$

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        reduce (str, optional): The reduce operation (`"sum"`, `"add"`, `"mean"`,
            `"min"` or `"max"`). Default is `"sum"`.

    Returns:
        paddle.Tensor, the reduced tensor.

    Examples:
        >>> from paddle_scatter import segment_csr

        >>> src = paddle.randn([10, 6, 64])
        >>> indptr = paddle.tensor([0, 2, 5, 6])
        >>> indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.

        >>> out = segment_csr(src, indptr, reduce="sum")

        >>> print(out.shape)
        [10, 3, 64]
    """
    if reduce == "sum" or reduce == "add":
        return segment_sum_csr(src, indptr, out)
    elif reduce == "mean":
        return segment_mean_csr(src, indptr, out)
    elif reduce == "min":
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == "max":
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError


def gather_csr(
    src: paddle.Tensor, indptr: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""Gather values from the `src` tensor into `out` at the
    indices specified within the ranges specified in the `indptr`
    tensor along the last dimension of `indptr`.

    Formally, if `src` and `indptr` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_1, ..., x_{m-1}, x_m, x_{m+1}, ..., x_n)` and
    :math:`(x_1, ..., x_{m-1}, y)` (y = x_m + 1), respectively,
    then `out` must be an :math:`n`-dimensional tensor with size
    :math:`(x_1, ..., x_{m-1}, k, x_{m+1}, ..., x_n)`, where :math:`k`
    is the number of segments specified by `indptr`.
    Moreover, the values of `indptr` must be between :math:`0` and
    :math:`x_m` in ascending order.
    The `indptr` tensor supports broadcasting in case its dimensions do
    not match with `src`.

    $$
    \mathrm{out}_i = \mathrm{src}[indptr[k]],
    k = indptr[(indptr - i <= 0)][-1]
    $$

    where :math:`i` is the index at the last dimension of `index`.

    Args:
        src (paddle.Tensor): The source tensor.
        indptr (paddle.Tensor): The index pointers between elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        paddle.Tensor, the gathered tensor.

    Examples:
        >>> from paddle_scatter import gather_csr

        >>> src = paddle.to_tensor([1, 2, 3, 4])
        >>> indptr = paddle.to_tensor([0, 2, 5, 5, 6])

        >>> out = gather_csr(src, indptr)

        >>> print(out)
        Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
        [1, 1, 2, 2, 2, 4])
    """
    indptr_shape = indptr.shape
    src_shape = src.shape
    dim = len(indptr_shape) - 1
    # broadcast indptr to src
    indptr_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        indptr = indptr.reshape(indptr_shape)
    else:
        indptr = indptr.expand(indptr_shape)
        assert (
            src_shape[dim] == indptr_shape[dim] - 1
        ), "The (size of indptr at last dimension) must be equal to\
                                                           the (size of src at the same dimension) + 1."
    if out is None:
        out_size = src_shape
        if indptr.numel() == 0:
            out_size[dim] = 0
        else:
            # refer to the original design in source cpp code
            out_size[dim] = indptr.flatten()[-1]
        out = zeros(out_size, dtype=src.dtype)
    else:
        out_size = out.shape
    if src.numel() == 0:
        return out

    repeats = indptr.diff(n=1, axis=dim)
    src_flatten = transform_3d(src, dim)
    out_flatten = transform_3d(out, dim)
    repeats_flatten = transform_2d(repeats, dim)
    for i in range(src_flatten.shape[0]):
        for j in range(src_flatten.shape[-1]):
            result = repeat_interleave(src_flatten[i, :, j], repeats_flatten[i], 0)
            repeat_sum = repeats_flatten[i].sum()
            if out_size[dim] >= repeat_sum:
                out_flatten[i, :repeat_sum, j] = result[:repeat_sum]
            else:
                out_flatten[i, :, j] = result[: out_size[dim]]
    assign(out_flatten.reshape(out_size), out)
    return out

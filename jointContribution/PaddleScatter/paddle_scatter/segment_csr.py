from typing import Optional
from typing import Tuple

import paddle
import paddle_scatter_ops


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

    if out is None:
        size = src.shape
        size[dim] = max(indptr_shape[dim] - 1, 0)
        if src.numel() == 0:
            return paddle.zeros(size, dtype=src.dtype)
        return paddle_scatter_ops.custom_segment_csr(src, indptr, None, size, "sum")[0]
    else:
        if src.numel() == 0:
            return out
        result = paddle_scatter_ops.custom_segment_csr(
            src, indptr, out, out.shape, "sum"
        )[0]
        paddle.assign(result, out)
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

    if out is None:
        size = src.shape
        size[dim] = max(indptr_shape[dim] - 1, 0)
        if src.numel() == 0:
            return paddle.zeros(size, dtype=src.dtype)
        return paddle_scatter_ops.custom_segment_csr(src, indptr, None, size, "mean")[0]
    else:
        if src.numel() == 0:
            return out
        result = paddle_scatter_ops.custom_segment_csr(
            src, indptr, out, out.shape, "mean"
        )[0]
        paddle.assign(result, out)
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
                paddle.zeros(size, dtype=src.dtype),
                paddle.full(size, src.shape[dim], indptr.dtype),
            )
        return paddle_scatter_ops.custom_segment_csr(src, indptr, None, size, "min")
    else:
        if src.numel() == 0:
            return (out, paddle.full(size, src.shape[dim], indptr.dtype))
        result, arg_result = paddle_scatter_ops.custom_segment_csr(
            src, indptr, out, out.shape, "min"
        )
        paddle.assign(result, out)
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

    size = src.shape
    if out is None:
        size[dim] = max(indptr_shape[dim] - 1, 0)
        if src.numel() == 0:
            return (
                paddle.zeros(size, dtype=src.dtype),
                paddle.full(size, src.shape[dim], indptr.dtype),
            )
        return paddle_scatter_ops.custom_segment_csr(src, indptr, None, size, "max")
    else:
        if src.numel() == 0:
            return (out, paddle.full(size, src.shape[dim], indptr.dtype))
        for i in range(len(size)):
            if i != dim:
                assert size[i] == out.shape[i]
        result, arg_result = paddle_scatter_ops.custom_segment_csr(
            src, indptr, out, out.shape, "max"
        )
        paddle.assign(result, out)
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
            return paddle.zeros(out_size, dtype=src.dtype)
        out_size[dim] = indptr.flatten()[-1]
        return paddle_scatter_ops.custom_gather_csr(src, indptr, None, out_size)
    else:
        if src.numel() == 0:
            return out
        out_size = out.shape
        for i in range(len(out_size)):
            if i != dim:
                assert src_shape[i] == out_size[i]
        result = paddle_scatter_ops.custom_gather_csr(src, indptr, out, out_size)
        paddle.assign(result, out)
        return out

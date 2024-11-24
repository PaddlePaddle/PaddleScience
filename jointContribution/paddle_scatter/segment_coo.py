from typing import Optional
from typing import Tuple

import paddle
from paddle import assign
from paddle import full
from paddle import slice
from paddle import take_along_axis
from paddle import to_tensor
from paddle import zeros
from paddle.geometric import segment_mean
from paddle.geometric import segment_sum
from paddle.nn.functional import pad
from paddle_scatter_min_max_ops import custom_segment_coo_min_max

from .utils import broadcast
from .utils import transform_3d


def segment_sum_coo(
    src: paddle.Tensor,
    index: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`. The reduction method is sum.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The sorted indices of elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by sum reduction method.
    """
    index_shape = index.shape
    dim = len(index_shape) - 1
    index = broadcast(index, src, dim)
    if out is None:
        out_size = src.shape
        if dim_size is not None:
            out_size[dim] = dim_size
        elif index.numel() == 0:
            out_size[dim] = 0
        else:
            tmp = index.index_select(
                index=to_tensor([index.shape[dim] - 1]), axis=dim
            ).squeeze(dim)
            tmp = tmp.max() if tmp.numel() > 1 else tmp
            out_size[dim] = int(tmp) + 1
        if src.numel() == 0:
            return zeros(out_size, dtype=src.dtype)
    else:
        if src.numel() == 0:
            return out
        out_size = out.shape
        init = out.clone()
    tmp = zeros(out_size, dtype=src.dtype)
    src_fatten = transform_3d(src, dim)
    index_flatten = transform_3d(index, dim)
    out_flatten = transform_3d(tmp, dim)
    for i in range(src_fatten.shape[0]):
        for j in range(src_fatten.shape[-1]):
            result = segment_sum(src_fatten[i, :, j], index_flatten[i, :, j])
            if out_size[dim] >= len(result):
                out_flatten[i, : len(result), j] = result
            else:
                out_flatten[i, :, j] = result[: out_size[dim]]
    res = out_flatten.reshape(out_size)
    if out is None:
        return res
    else:
        res = res + init
        assign(res, out)
    return out


def segment_add_coo(
    src: paddle.Tensor,
    index: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`. The reduction method is sum.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The sorted indices of elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by sum reduction method.
    """
    return segment_sum_coo(src, index, out, dim_size)


def segment_mean_coo(
    src: paddle.Tensor,
    index: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`. The reduction method is mean.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The sorted indices of elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        paddle.Tensor, the reduced tensor by mean reduction method.
    """
    index_shape = index.shape
    dim = len(index_shape) - 1
    index = broadcast(index, src, dim)
    if out is None:
        out_size = src.shape
        if dim_size is not None:
            out_size[dim] = dim_size
        elif index.numel() == 0:
            out_size[dim] = 0
        else:
            tmp = index.index_select(
                index=to_tensor([index.shape[dim] - 1]), axis=dim
            ).squeeze(dim)
            tmp = tmp.max() if tmp.numel() > 1 else tmp
            out_size[dim] = int(tmp) + 1
        out = zeros(out_size, dtype=src.dtype)
    else:
        out_size = out.shape
    if src.numel() == 0:
        return out
    src_fatten = transform_3d(src, dim)
    index_flatten = transform_3d(index, dim)
    out_flatten = transform_3d(out, dim)
    for i in range(src_fatten.shape[0]):
        for j in range(src_fatten.shape[-1]):
            result = segment_mean(src_fatten[i, :, j], index_flatten[i, :, j])
            if out_size[dim] >= len(result):
                out_flatten[i, : len(result), j] = result
            else:
                out_flatten[i, :, j] = result[: out_size[dim]]
    assign(out_flatten.reshape(out_size), out)
    return out


def segment_min_coo(
    src: paddle.Tensor,
    index: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`. The reduction method is min.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The sorted indices of elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor], the reduced min tensor and arg_min tensor.
    """
    src_shape = src.shape
    index_shape = index.shape
    dim = len(index_shape) - 1
    # broadcast indptr to src
    index_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        index = index.reshape(index_shape)
    else:
        index = index.expand(index_shape)
    size = src.shape
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        tmp = index.index_select(
            index=to_tensor([index.shape[dim] - 1]), axis=dim
        ).squeeze(dim)
        tmp = tmp.max() if tmp.numel() > 1 else tmp
        size[dim] = int(tmp) + 1

    if out is None:
        if src.numel() == 0:
            return (
                zeros(size, dtype=src.dtype),
                full(size, src.shape[dim], index.dtype),
            )
        return custom_segment_coo_min_max(src, index, None, size, "min")
    else:
        if src.numel() == 0:
            return (out, full(size, src.shape[dim], index.dtype))
        for i in range(len(size)):
            if i != dim:
                assert size[i] == out.shape[i]
        result, arg_result = custom_segment_coo_min_max(
            src, index, out, out.shape, "min"
        )
        assign(result, out)
        return out, arg_result


def segment_max_coo(
    src: paddle.Tensor,
    index: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`. The reduction method is max.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The sorted indices of elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor], the reduced max tensor and arg_max tensor.
    """
    src_shape = src.shape
    index_shape = index.shape
    dim = len(index_shape) - 1
    # broadcast indptr to src
    index_shape[:dim] = src_shape[:dim]
    if src.numel() == 0:
        index = index.reshape(index_shape)
    else:
        index = index.expand(index_shape)
    size = src.shape
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        tmp = index.index_select(
            index=to_tensor([index.shape[dim] - 1]), axis=dim
        ).squeeze(dim)
        tmp = tmp.max() if tmp.numel() > 1 else tmp
        size[dim] = int(tmp) + 1

    if out is None:
        if src.numel() == 0:
            return (
                zeros(size, dtype=src.dtype),
                full(size, src.shape[dim], index.dtype),
            )
        return custom_segment_coo_min_max(src, index, None, size, "max")
    else:
        if src.numel() == 0:
            return (out, full(size, src.shape[dim], index.dtype))
        for i in range(len(size)):
            if i != dim:
                assert size[i] == out.shape[i]
        result, arg_result = custom_segment_coo_min_max(
            src, index, out, out.shape, "max"
        )
        assign(result, out)
        return out, arg_result


def segment_coo(
    src: paddle.Tensor,
    index: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    r"""Reduces all values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`.

    For each value in `src`, its output index is specified by its index
    in `src` for dimensions outside of `index.dim() - 1` and by the
    corresponding value in `index` for dimension `index.dim() - 1`.
    The applied reduction is defined via the `reduce` argument.

    Formally, if `src` and `index` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_1, ..., x_{m-1}, x_m, x_{m+1}, ..., x_n)` and
    :math:`(x_1, ..., x_{m-1}, x_m)`, respectively, then `out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_1, ..., x_{m-1}, y, x_{m+1}, ..., x_n)`.
    Moreover, the values of `index` must be between :math:`0` and
    :math:`y - 1` in ascending order.
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
        In contrast to :meth:`scatter`, this method expects values in `index`
        **to be sorted** along dimension `index.dim() - 1`.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The sorted indices of elements to segment.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.
        dim_size (int|None, optional): If `out` is not given, automatically create output
            with size `dim_size` at dimension `dim`. If `dim_size` is not given,
            a minimal sized output tensor according to `index.max() + 1` is returned.
            Default is None.
        reduce (str, optional): The reduce operation (`"sum"`, `"add"`, `"mean"`,
            `"min"` or `"max"`). Default is `"sum"`.

    Returns:
        paddle.Tensor, the reduced tensor.

    Examples:
        >>> from paddle_scatter import segment_coo

        >>> src = paddle.randn([10, 6, 64])
        >>> index = paddle.to_tensor([0, 0, 1, 1, 1, 2])
        >>> index = index.view(1, -1)  # Broadcasting in the first and last dim.

        >>> out = segment_coo(src, index, reduce="sum")

        >>> print(out.shape)
        [10, 3, 64]
    """
    if reduce == "sum" or reduce == "add":
        return segment_sum_coo(src, index, out, dim_size)
    elif reduce == "mean":
        return segment_mean_coo(src, index, out, dim_size)
    elif reduce == "min":
        return segment_min_coo(src, index, out, dim_size)[0]
    elif reduce == "max":
        return segment_max_coo(src, index, out, dim_size)[0]
    else:
        raise ValueError


def gather_coo(
    src: paddle.Tensor, index: paddle.Tensor, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""Gather values from the `src` tensor into `out` at the
    indices specified in the `index` tensor along the last dimension of
    `index`.

    Formally, if `src` and `index` are :math:`n`-dimensional and
    :math:`m`-dimensional tensors with
    size :math:`(x_1, ..., x_{m-1}, x_m, x_{m+1}, ..., x_n)` and
    :math:`(x_1, ..., x_{m-1}, y)`, respectively, then `out` must be an
    :math:`n`-dimensional tensor with size
    :math:`(x_1, ..., x_{m-1}, y, x_{m+1}, ..., x_n)`.
    Moreover, the elements of `index` must be between :math:`0` and
    :math:`x_m - 1` in ascending order.
    The `index` tensor supports broadcasting in case its dimensions do
    not match with `src`.

    $$
    \mathrm{out}_{i} = \mathrm{src}_{\mathrm{index}_{i}}
    $$

    where :math:`i` is the index at the last dimension of `index`.

    Args:
        src (paddle.Tensor): The source tensor.
        index (paddle.Tensor): The indices of elements to gather.
            The number of dimensions of `index` needs to be less than or
            equal to `src`.
        out (paddle.Tensor|None, optional): The destination tensor. Default is None.

    Returns:
        paddle.Tensor, the gathered tensor.

    Examples:
        >>> from paddle_scatter import gather_coo

        >>> src = paddle.to_tensor([1, 2, 3, 4])
        >>> index = paddle.to_tensor([0, 0, 1, 1, 1, 3])

        >>> out = gather_coo(src, index)

        >>> print(out)
        Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
        [1, 1, 2, 2, 2, 4])
    """
    index_shape = index.shape
    dim = len(index_shape) - 1
    src_shape = src.shape

    # broadcast index's dimension to the same as src's
    for _ in range(len(src_shape) - len(index_shape)):
        index = index.unsqueeze(-1)
    new_index_shape = src_shape.copy()
    new_index_shape[dim] = index_shape[dim]
    if src.numel() == 0:
        index = index.reshape(new_index_shape)
    else:
        index = index.expand(new_index_shape)

    if out is None:
        out_size = src_shape
        if index.numel() == 0:
            out_size[dim] = 0
        else:
            out_size[dim] = index.shape[dim]
        out = zeros(out_size, dtype=src.dtype)
    else:
        out_size = out.shape
    if src.numel() == 0:
        return out

    result = take_along_axis(src, index, dim, broadcast=False)
    if out_size[dim] > result.shape[dim]:
        padding = [0] * 2 * len(out_size)
        padding[2 * dim + 1] = out_size[dim] - result.shape[dim]
        result = pad(result, padding, mode="constant", value=0)
    elif out_size[dim] < result.shape[dim]:
        result = slice(result, [dim], 0, out_size[dim])
    return assign(result, out)

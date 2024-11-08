from logging import Logger
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from .types import Array
from .types import Device
from .types import Scalar


def as_tensor(
    arg: Union[Scalar, Array],
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Create tensor from array if argument is not of type paddle.Tensor.

    Unlike ``paddle.as_tensor()``, this function preserves the tensor device if ``device=None``.

    """
    if isinstance(arg, paddle.Tensor):
        return arg
    if device is None and isinstance(arg, paddle.Tensor):
        device = arg.place
    return paddle.to_tensor(data=arg, dtype=dtype, place=device)


def as_float_tensor(arr: Array) -> paddle.Tensor:
    """Create tensor with floating point type from argument if it is not yet."""
    arr_ = as_tensor(arr)
    if not paddle.is_floating_point(x=arr_):
        return arr_.astype("float32")
    return arr_


def as_one_hot_tensor(
    tensor: paddle.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    """Converts label image to one-hot encoding of multi-class segmentation.

    Adapted from: https://github.com/wolny/paddle-3dunet

    Args:
        tensor: Input tensor of shape ``(N, 1, ..., X)`` or ``(N, C, ..., X)``.
            When a tensor with ``C == num_classes`` is given, it is converted to the specified
            ``dtype`` but not modified otherwise. Otherwise, the input tensor must contain
            class labels in a single channel.
        num_classes: Number of channels/labels.
        ignore_index: Ignore index to be kept during the expansion. The locations of the index
            value in the GT image is stored in the corresponding locations across all channels so
            that this location can be ignored across all channels later e.g. in Dice computation.
            This argument must be ``None`` if ``tensor`` has ``C == num_channels``.
        dtype: Data type of output tensor. Default is ``paddle.float``.

    Returns:
        Output tensor of shape ``(N, C, ..., X)``.

    """
    if dtype is None:
        dtype = "float32"
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError("as_one_hot_tensor() 'tensor' must be paddle.Tensor")
    if tensor.dim() < 3:
        raise ValueError("as_one_hot_tensor() 'tensor' must have shape (N, C, ..., X)")
    if tuple(tensor.shape)[1] == num_classes:
        return tensor.to(dtype=dtype)
    elif tuple(tensor.shape)[1] != 1:
        raise ValueError(
            f"as_one_hot_tensor() 'tensor' must have shape (N, 1|{num_classes}, ..., X)"
        )
    shape = list(tuple(tensor.shape))
    shape[1] = num_classes
    if ignore_index is None:
        return (
            paddle.zeros(shape=shape, dtype=dtype)
            .to(tensor.place)
            .put_along_axis_(axis=1, indices=tensor, values=1.0)
        )
    mask = tensor.expand(shape=shape) == ignore_index
    inputs = tensor.clone()
    inputs[inputs == ignore_index] = 0
    result = (
        paddle.zeros(shape=shape, dtype=dtype)
        .to(inputs.place)
        .put_along_axis_(axis=1, indices=inputs, values=1.0)
    )
    result[mask] = ignore_index
    return result


def atleast_1d(
    arr: Array, dtype: Optional[paddle.dtype] = None, device: Optional[Device] = None
) -> paddle.Tensor:
    """Convert array-like argument to 1- or more-dimensional tensor."""
    arr_ = as_tensor(arr, dtype=dtype, device=device)
    return arr_.unsqueeze(axis=0) if arr_.ndim == 0 else arr_


def cat_scalars(
    arg: Union[Scalar, Array],
    *args: Scalar,
    num: int = 0,
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Device] = None,
) -> paddle.Tensor:
    """Join arguments into single 1-dimensional tensor.

    This auxiliary function is used by ``Grid``, ``Image``, and ``ImageBatch`` to support
    method arguments for different spatial dimensions as either scalar constant, list
    of scalar ``*args``, or single ``Array`` argument. If a single argument of type ``Array``
    is given, it must be a sequence of scalar values.

    Args:
        arg: Either a single scalar or sequence of scalars. If the argument is a ``paddle.Tensor``,
            it is cloned and detached in order to avoid unintended side effects.
        args: Additional scalars. If ``arg`` is a sequence, ``args`` must be empty.
        num: Number of expected scalar values. If a single scalar ``arg`` is given,
            it is repeated ``num`` times to create a 1-dimensional array. If ``num=0``,
            the length of the returned array corresponds to the number of given scalars.
        dtype: Data type of output tensor.
        device: Device on which to store tensor.

    Returns:
        Scalar arguments joined into a 1-dimensional tensor.

    """
    if args:
        if isinstance(arg, (tuple, list)) or isinstance(arg, paddle.Tensor):
            raise ValueError("arg and args must either be all scalars, or args empty")
        arg = paddle.to_tensor(data=(arg,) + args, dtype=dtype, place=device)
    else:
        arg = as_tensor(arg, dtype=dtype, device=device)
    if arg.ndim == 0:
        arg = arg.unsqueeze(0)
    if arg.ndim != 1:
        if num > 0:
            raise ValueError(
                f"Expected one scalar, a sequence of length {num}, or {num} args"
            )
        raise ValueError(
            "Expected one scalar, a sequence of scalars, or multiple scalars"
        )
    if num > 0:
        if len(arg) == 1:
            arg = arg.repeat(num)
        elif len(arg) != num:
            raise ValueError(
                f"Expected one scalar, a sequence of length {num}, or {num} args"
            )
    return arg


def batched_index_select(
    input: paddle.Tensor, dim: int, index: paddle.Tensor
) -> paddle.Tensor:
    """Batched version of paddle.index_select().

    See https://discuss.paddle.org/t/batched-index-select/9115/9.

    """
    for i in range(1, len(tuple(input.shape))):
        if i != dim:
            index = index.unsqueeze(axis=i)
    shape = list(tuple(input.shape))
    shape[0] = -1
    shape[dim] = -1
    index = index.expand(shape=shape)
    return paddle.take_along_axis(arr=input, axis=dim, indices=index)


def move_dim(tensor: paddle.Tensor, dim: int, pos: int) -> paddle.Tensor:
    """Move the specified tensor dimension to another position."""
    if dim < 0:
        dim = tensor.ndim + dim
    if pos < 0:
        pos = tensor.ndim + pos
    if pos == dim:
        return tensor
    if dim < pos:
        pos += 1
    tensor = tensor.unsqueeze(axis=pos)
    if pos <= dim:
        dim += 1
    x = tensor
    perm_6 = list(range(x.ndim))
    perm_6[dim] = pos
    perm_6[pos] = dim
    tensor = x.transpose(perm=perm_6).squeeze(axis=dim)
    return tensor


def unravel_coords(indices: paddle.Tensor, size: Tuple[int, ...]) -> paddle.Tensor:
    """Converts flat indices into unraveled grid coordinates.

    Args:
        indices: A tensor of flat indices with shape ``(..., N)``.
        size: Sampling grid size with order ``(X, ...)``.

    Returns:
        Grid coordinates of corresponding grid points.

    """
    size = tuple(size)
    numel = size.size
    if indices.greater_equal(y=paddle.to_tensor(numel)).astype("bool").any():
        raise ValueError(f"unravel_coords() indices must be smaller than {numel}")
    coords = paddle.zeros(
        shape=tuple(indices.shape) + (len(size),), dtype=indices.dtype
    )
    for i, n in enumerate(size):
        coords[..., i] = indices % n
        indices = indices // n
    return coords


def unravel_index(indices: paddle.Tensor, shape: Tuple[int, ...]) -> paddle.Tensor:
    """Converts flat indices into unraveled coordinates in a target shape.

    This is a `paddle` implementation of `numpy.unravel_index`, but returning a
    tensor of shape (..., N, D) rather than a D-dimensional tuple. See also

    Args:
        indices: A tensor of indices with shape (..., N).
        shape: The targeted tensor shape of length D.

    Returns:
        Unraveled coordinates as tensor of shape (..., N, D) with coordinates
        in the same order as the input ``shape`` dimensions.

    """
    shape = tuple(shape)
    numel = shape.size
    if indices.greater_equal(y=paddle.to_tensor(numel)).astype("bool").any():
        raise ValueError(f"unravel_coords() indices must be smaller than {numel}")
    coords = paddle.zeros(
        shape=tuple(indices.shape) + (len(shape),), dtype=indices.dtype
    )
    for i, n in enumerate(reversed(shape)):
        coords[..., i] = indices % n
        indices = indices // n
    return coords.flip(axis=-1)


def log_grad_hook(
    name: str, logger: Optional[Logger] = None
) -> Callable[[paddle.Tensor], None]:
    """Backward hook to print tensor gradient information for debugging."""

    def printer(grad: paddle.Tensor) -> None:
        if grad.size == 1:
            msg = f"{name}.grad: value={grad}"
        else:
            msg = f"{name}.grad: shape={tuple(tuple(grad.shape))}, max={grad.max()}, min={grad.min()}, mean={grad.mean()}"
        if logger is None:
            print(msg)
        else:
            logger.debug(msg)

    return printer


def register_backward_hook(
    tensor: paddle.Tensor,
    hook: Callable[[paddle.Tensor], None],
    retain_grad: bool = False,
) -> paddle.Tensor:
    """Register backward hook and optionally enable retaining gradient."""
    if not tensor.stop_gradient:
        if retain_grad:
            tensor.retain_grads()
        tensor.register_hook(hook=hook)
    return tensor

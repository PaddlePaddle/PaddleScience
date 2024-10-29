from collections import namedtuple
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import overload

import deepali.utils.paddle_aux  # noqa
import paddle
from paddle import Tensor

from .typing import ScalarOrTuple


def get_namedtuple_item(self: NamedTuple, arg: Union[int, str]) -> Any:
    if isinstance(arg, str):
        return getattr(self, arg)
    return self[arg]


def namedtuple_keys(self: NamedTuple) -> Iterable[str]:
    return self._fields


def namedtuple_values(self: NamedTuple) -> Iterable[Any]:
    return self


def namedtuple_items(self: NamedTuple) -> Iterable[Tuple[str, Any]]:
    return zip(self._fields, self)


def as_immutable_container(
    arg: Union[paddle.Tensor, Sequence, Mapping], recursive: bool = True
) -> Union[paddle.Tensor, tuple]:
    r"""Convert mutable container such as dict or list to an immutable container type.

    For use with ``paddle.utils.tensorboard.SummaryWriter.add_graph`` when model output is list or dict.
    See error message: "Encountering a dict at the output of the tracer might cause the trace to be incorrect,
    this is only valid if the container structure does not change based on the module's inputs. Consider using
    a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead).
    If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior."

    """
    if recursive:
        if isinstance(arg, Mapping):
            arg = {key: as_immutable_container(value) for key, value in arg.items()}
        elif isinstance(arg, Sequence):
            arg = [as_immutable_container(value) for value in arg]
    if isinstance(arg, Mapping):
        output_type = namedtuple("Dict", sorted(arg.keys()))
        output_type.__getitem__ = get_namedtuple_item
        output_type.keys = namedtuple_keys
        output_type.values = namedtuple_values
        output_type.items = namedtuple_items
        return output_type(**arg)
    if isinstance(arg, list):
        return tuple(arg)
    return arg


def conv_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
    dilation: ScalarOrTuple[int] = 1,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after convolution."""
    device = paddle.CPUPlace()
    m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=stride, dtype="int32", place=device))
    d: Tensor = paddle.atleast_1d(paddle.to_tensor(data=dilation, dtype="int32", place=device))
    if m.ndim != 1:
        raise ValueError("conv_output_size() 'in_size' must be scalar or sequence")
    ndim = tuple(m.shape)[0]
    if ndim == 1 and tuple(k.shape)[0] > 1:
        ndim = tuple(k.shape)[0]
    for arg, name in zip([k, s, d], ["kernel_size", "stride", "dilation"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"conv_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(shape=ndim)
    s = s.expand(shape=ndim)
    d = d.expand(shape=ndim)
    if padding == "valid":
        padding = 0
    elif padding == "same":
        if not s.equal(y=1).astype("bool").all():
            raise ValueError("conv_output_size() padding='same' requires stride=1")
        padding = same_padding(kernel_size=kernel_size, dilation=dilation)
    elif isinstance(padding, str):
        raise ValueError("conv_output_size() 'padding' string must be 'valid' or 'same'")
    p: Tensor = paddle.atleast_1d(paddle.to_tensor(data=padding, dtype="int32", place=device))
    if p.ndim != 1 or tuple(p.shape)[0] not in (1, ndim):
        raise ValueError(
            f"conv_output_size() 'padding' must be scalar or sequence of length {ndim}"
        )
    p = p.expand(shape=ndim)
    n = (
        p.mul(2)
        .add_(y=paddle.to_tensor(m))
        .subtract_(y=paddle.to_tensor(k.sub(1).multiply_(y=paddle.to_tensor(d))))
        .subtract_(y=paddle.to_tensor(1))
        .astype(dtype="float32")
        .divide_(y=paddle.to_tensor(s))
        .add_(y=paddle.to_tensor(1))
        .floor_()
        .astype(dtype="int64")
    )
    if isinstance(in_size, int):
        return n[0].item()
    return tuple(n.tolist())


def conv_transposed_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
    output_padding: ScalarOrTuple[int] = 0,
    dilation: ScalarOrTuple[int] = 1,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after transposed convolution."""
    device = paddle.CPUPlace()
    m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=stride, dtype="int32", place=device))
    p: Tensor = paddle.atleast_1d(paddle.to_tensor(data=padding, dtype="int32", place=device))
    o: Tensor = paddle.atleast_1d(
        paddle.to_tensor(data=output_padding, dtype="int32", place=device)
    )
    d: Tensor = paddle.atleast_1d(paddle.to_tensor(data=dilation, dtype="int32", place=device))
    if m.ndim != 1:
        raise ValueError("conv_transposed_output_size() 'in_size' must be scalar or sequence")
    ndim = tuple(m.shape)[0]
    if ndim == 1 and tuple(k.shape)[0] > 1:
        ndim = tuple(k.shape)[0]
    for arg, name in zip(
        [k, s, p, o, d], ["kernel_size", "stride", "padding", "output_padding", "dilation"]
    ):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"conv_transposed_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(shape=ndim)
    s = s.expand(shape=ndim)
    p = p.expand(shape=ndim)
    o = o.expand(shape=ndim)
    d = d.expand(shape=ndim)
    n = (
        m.sub(1)
        .multiply_(y=paddle.to_tensor(s))
        .subtract_(y=paddle.to_tensor(p.mul(2)))
        .add_(y=paddle.to_tensor(k.sub(1).multiply_(y=paddle.to_tensor(d))))
        .add_(y=paddle.to_tensor(o))
        .add_(y=paddle.to_tensor(1))
    )
    if isinstance(in_size, int):
        return n.item()
    return tuple(n.tolist())


def pad_output_size(
    in_size: ScalarOrTuple[int], padding: ScalarOrTuple[int] = 0
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after padding."""
    device = paddle.CPUPlace()
    m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
    p: Tensor = paddle.atleast_1d(paddle.to_tensor(data=padding, dtype="int32", place=device))
    if m.ndim != 1:
        raise ValueError("pad_output_size() 'in_size' must be scalar or sequence")
    ndim = tuple(m.shape)[0]
    if ndim == 1 and tuple(p.shape)[0] > 1 and tuple(p.shape)[0] % 2:
        ndim = tuple(p.shape)[0] // 2
    if p.ndim != 1 or tuple(p.shape)[0] not in (1, 2 * ndim):
        raise ValueError(
            f"pad_output_size() 'padding' must be scalar or sequence of length {2 * ndim}"
        )
    p = p.expand(shape=2 * ndim)
    n = p.reshape(ndim, 2).sum(axis=1).add(m)
    if isinstance(in_size, int):
        return n[0].item()
    return tuple(n.tolist())


def pool_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
    dilation: ScalarOrTuple[int] = 1,
    ceil_mode: bool = False,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after pooling."""
    device = paddle.CPUPlace()
    m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=stride, dtype="int32", place=device))
    p: Tensor = paddle.atleast_1d(paddle.to_tensor(data=padding, dtype="int32", place=device))
    d: Tensor = paddle.atleast_1d(paddle.to_tensor(data=dilation, dtype="int32", place=device))
    if m.ndim != 1:
        raise ValueError("pool_output_size() 'in_size' must be scalar or sequence")
    ndim = tuple(m.shape)[0]
    if ndim == 1 and tuple(k.shape)[0] > 1:
        ndim = tuple(k.shape)[0]
    for arg, name in zip([k, s, p, d], ["kernel_size", "stride", "padding", "dilation"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"pool_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(shape=ndim)
    s = s.expand(shape=ndim)
    p = p.expand(shape=ndim)
    d = d.expand(shape=ndim)
    n = (
        p.mul(2)
        .add_(y=paddle.to_tensor(m))
        .subtract_(y=paddle.to_tensor(k.sub(1).multiply_(y=paddle.to_tensor(d))))
        .subtract_(y=paddle.to_tensor(1))
        .astype(dtype="float32")
        .divide_(y=paddle.to_tensor(s))
        .add_(y=paddle.to_tensor(1))
    )
    n = n.ceil() if ceil_mode else n.floor()
    n = n.astype(dtype="int64")
    if isinstance(in_size, int):
        return n[0].item()
    return tuple(n.tolist())


def unpool_output_size(
    in_size: ScalarOrTuple[int],
    kernel_size: ScalarOrTuple[int],
    stride: ScalarOrTuple[int] = 1,
    padding: ScalarOrTuple[int] = 0,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after unpooling."""
    device = paddle.CPUPlace()
    m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=stride, dtype="int32", place=device))
    p: Tensor = paddle.atleast_1d(paddle.to_tensor(data=padding, dtype="int32", place=device))
    if m.ndim != 1:
        raise ValueError("unpool_output_size() 'in_size' must be scalar or sequence")
    ndim = tuple(m.shape)[0]
    if ndim == 1 and tuple(k.shape)[0] > 1:
        ndim = tuple(k.shape)[0]
    for arg, name in zip([k, s, p], ["kernel_size", "stride", "padding"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"unpool_output_size() {name!r} must be scalar or sequence of length {ndim}"
            )
    k = k.expand(shape=ndim)
    s = s.expand(shape=ndim)
    p = p.expand(shape=ndim)
    n = m.sub(1).multiply_(y=paddle.to_tensor(s)).subtract_(y=paddle.to_tensor(p.mul(2))).add(k)
    if isinstance(in_size, int):
        return n[0].item()
    return tuple(n.tolist())


@overload
def same_padding(kernel_size: int, dilation: int = 1) -> int:
    ...


@overload
def same_padding(kernel_size: Tuple[int, ...], dilation: int = 1) -> Tuple[int, ...]:
    ...


@overload
def same_padding(kernel_size: int, dilation: Tuple[int, ...]) -> Tuple[int, ...]:
    ...


@overload
def same_padding(kernel_size: Tuple[int, ...], dilation: Tuple[int, ...]) -> Tuple[int, ...]:
    ...


def same_padding(
    kernel_size: ScalarOrTuple[int], dilation: ScalarOrTuple[int] = 1
) -> ScalarOrTuple[int]:
    r"""Padding value needed to ensure convolution preserves input tensor shape.

    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``(kernel_size - 1) * dilation`` is an odd number.

    """
    # Adapted from Project MONAI
    # https://github.com/Project-MONAI/MONAI/blob/db8f7877da06a9b3710071c626c0488676716be1/monai/networks/layers/convutils.py
    device = paddle.CPUPlace()
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    d: Tensor = paddle.atleast_1d(paddle.to_tensor(data=dilation, dtype="int32", place=device))
    if k.ndim != 1:
        raise ValueError("same_padding() 'kernel_size' must be scalar or sequence")
    ndim = tuple(k.shape)[0]
    if ndim == 1 and tuple(d.shape)[0] > 1:
        ndim = tuple(d.shape)[0]
    for arg, name in zip([k, d], ["kernel_size", "dilation"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(f"same_padding() {name!r} must be scalar or sequence of length {ndim}")
    if k.sub(1).mul(d).mod(y=paddle.to_tensor(2)).equal(y=1).astype("bool").any():
        raise NotImplementedError(
            f"Same padding not available for kernel_size={tuple(k.tolist())} and dilation={tuple(d.tolist())}."
        )
    p = k.sub(1).div(2).mul(d.astype("float32")).astype("int32")
    if isinstance(kernel_size, int) and isinstance(dilation, int):
        return p[0].item()
    return tuple(p.tolist())


@overload
def stride_minus_kernel_padding(kernel_size: int, stride: int) -> int:
    ...


@overload
def stride_minus_kernel_padding(kernel_size: Sequence[int], stride: int) -> Tuple[int, ...]:
    ...


@overload
def stride_minus_kernel_padding(kernel_size: int, stride: Sequence[int]) -> Tuple[int, ...]:
    ...


def stride_minus_kernel_padding(
    kernel_size: ScalarOrTuple[int], stride: ScalarOrTuple[int]
) -> ScalarOrTuple[int]:
    # Adapted from Project MONAI
    # https://github.com/Project-MONAI/MONAI/blob/db8f7877da06a9b3710071c626c0488676716be1/monai/networks/layers/convutils.py
    device = paddle.CPUPlace()
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=stride, dtype="int32", place=device))
    if k.ndim != 1:
        raise ValueError("stride_minus_kernel_padding() 'kernel_size' must be scalar or sequence")
    ndim = tuple(k.shape)[0]
    if ndim == 1 and tuple(s.shape)[0] > 1:
        ndim = tuple(s.shape)[0]
    for arg, name in zip([k, s], ["kernel_size", "stride"]):
        if arg.ndim != 1 or arg.shape[0] not in (1, ndim):
            raise ValueError(
                f"stride_minus_kernel_padding() {name!r} must be scalar or sequence of length {ndim}"
            )
    assert k.ndim == 1, "stride_minus_kernel_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "stride_minus_kernel_padding() 'stride' must be scalar or sequence"
    p = s.sub(k).astype("int32")
    if isinstance(kernel_size, int) and isinstance(stride, int):
        return p[0].item()
    return tuple(p.tolist())


def upsample_padding(
    kernel_size: ScalarOrTuple[int], scale_factor: ScalarOrTuple[int]
) -> Tuple[int, ...]:
    r"""Padding on both sides for transposed convolution."""
    device = paddle.CPUPlace()
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=scale_factor, dtype="int32", place=device))
    assert k.ndim == 1, "upsample_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "upsample_padding() 'scale_factor' must be scalar or sequence"
    p = k.sub(s).add(1).div(2).astype("int32")
    if p.less_than(y=paddle.to_tensor(0)).astype("bool").any():
        raise ValueError(
            "upsample_padding() 'kernel_size' must be greater than or equal to 'scale_factor'"
        )
    return tuple(p.tolist())


def upsample_output_padding(
    kernel_size: ScalarOrTuple[int], scale_factor: ScalarOrTuple[int], padding: ScalarOrTuple[int]
) -> Tuple[int, ...]:
    r"""Output padding on one side for transposed convolution."""
    device = paddle.CPUPlace()
    k: Tensor = paddle.atleast_1d(paddle.to_tensor(data=kernel_size, dtype="int32", place=device))
    s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=scale_factor, dtype="int32", place=device))
    p: Tensor = paddle.atleast_1d(paddle.to_tensor(data=padding, dtype="int32", place=device))
    assert k.ndim == 1, "upsample_output_padding() 'kernel_size' must be scalar or sequence"
    assert s.ndim == 1, "upsample_output_padding() 'scale_factor' must be scalar or sequence"
    assert p.ndim == 1, "upsample_output_padding() 'padding' must be scalar or sequence"
    op = p.mul(2).sub(k).add(s).astype("int32")
    if op.less_than(y=paddle.to_tensor(0)).astype("bool").any():
        raise ValueError(
            "upsample_output_padding() 'output_padding' must be greater than or equal to zero"
        )
    return tuple(op.tolist())


def upsample_output_size(
    in_size: ScalarOrTuple[int],
    size: Optional[ScalarOrTuple[int]] = None,
    scale_factor: Optional[ScalarOrTuple[float]] = None,
) -> ScalarOrTuple[int]:
    r"""Calculate spatial size of output tensor after unpooling."""
    if size is not None and scale_factor is not None:
        raise ValueError("upsample_output_size() 'size' and 'scale_factor' are mutually exclusive")
    device = paddle.CPUPlace()
    m: Tensor = paddle.atleast_1d(paddle.to_tensor(data=in_size, dtype="int32", place=device))
    if m.ndim != 1:
        raise ValueError("upsample_output_size() 'in_size' must be scalar or sequence")
    ndim = tuple(m.shape)[0]
    if size is not None:
        s: Tensor = paddle.atleast_1d(paddle.to_tensor(data=size, dtype="int32", place=device))
        if s.ndim != 1 or tuple(s.shape)[0] not in (1, ndim):
            raise ValueError(
                f"upsample_output_size() 'size' must be scalar or sequence of length {ndim}"
            )
        n = s.expand(shape=ndim)
    elif scale_factor is not None:
        s: Tensor = paddle.atleast_1d(
            paddle.to_tensor(data=scale_factor, dtype="int32", place=device)
        )
        if s.ndim != 1 or tuple(s.shape)[0] not in (1, ndim):
            raise ValueError(
                f"upsample_output_size() 'scale_factor' must be scalar or sequence of length {ndim}"
            )
        n = m.astype(dtype="float32").mul(s).floor().astype(dtype="int64")
    else:
        n = m
    if isinstance(in_size, int):
        return n.item()
    return tuple(n.tolist())

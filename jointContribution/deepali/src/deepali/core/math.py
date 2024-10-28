r"""Basic math operations with tensors."""

from typing import Optional
from typing import Union

import paddle
from deepali.utils import paddle_aux  # noqa

from .typing import Scalar


def abspow(x: paddle.Tensor, exponent: Union[int, float]) -> paddle.Tensor:
    r"""Compute ``abs(x)**exponent``."""
    if exponent == 1:
        return x.abs()
    return x.abs().pow(y=exponent)


def atanh(x: paddle.Tensor) -> paddle.Tensor:
    r"""Inverse of tanh function.

    Args:
        x: Function argument.

    Returns:
        Inverse of tanh function, i.e., ``y`` where ``x = tanh(y)``.

    See also:
        https://github.com/pytorch/pytorch/issues/10324

    """
    return paddle.log1p(x=2 * x / (1 - x)) / 2


def max_difference(source: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    r"""Maximum possible intensity difference.

    Note that the two input images need not be sampled on the same grid.

    Args:
        source: Source image.
        target: Reference image.

    Returns:
        Maximum possible intensity difference.

    """
    smin, smax = source.min(), source.max()
    if target is source:
        tmin, tmax = smin, smax
    else:
        tmin, tmax = target.min(), target.max()
    return paddle_aux.max(paddle.abs(x=smax - tmin), paddle.abs(x=tmax - smin))


def round_decimals(
    tensor: paddle.Tensor, decimals: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""Round tensor values to specified number of decimals."""
    if not decimals:
        result = paddle.assign(paddle.round(tensor), output=out)
    else:
        scale = 10**decimals
        if out is tensor:
            tensor *= scale
        else:
            tensor = tensor * scale
        result = paddle.assign(paddle.round(tensor), output=out)
        result /= scale
    return result


def threshold(
    data: paddle.Tensor, min: Optional[Scalar], max: Optional[Scalar] = None
) -> paddle.Tensor:
    r"""Get mask for given lower and upper thresholds.

    Args:
        data: Input data tensor.
        min: Lower threshold. If ``None``, use ``data.min()``.
        max: Upper threshold. If ``None``, use ``data.max()``.

    Returns:
        Boolean tensor with same shape as ``data``, where only elements with a value
        greater than or equal ``min`` and less than or equal ``max`` are ``True``.

    """
    if min is None and max is None:
        return paddle.ones_like(x=data, dtype="bool")
    if min is None:
        return data <= max
    if max is None:
        return data >= min
    return (min <= data) & (data <= max)

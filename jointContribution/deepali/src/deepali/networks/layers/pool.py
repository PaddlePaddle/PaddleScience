r"""Pooling layers."""

from functools import partial
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import paddle
from paddle import Tensor

from .lambd import LambdaLayer

PoolFunc = Callable[[Tensor], paddle.Tensor]
PoolArg = Union[PoolFunc, str, Mapping, Sequence, None]
POOLING_TYPES = {
    "avg": (paddle.nn.AvgPool1D, paddle.nn.AvgPool2D, paddle.nn.AvgPool3D),
    "avgpool": (paddle.nn.AvgPool1D, paddle.nn.AvgPool2D, paddle.nn.AvgPool3D),
    "adaptiveavg": (
        paddle.nn.AdaptiveAvgPool1D,
        paddle.nn.AdaptiveAvgPool2D,
        paddle.nn.AdaptiveAvgPool3D,
    ),
    "adaptiveavgpool": (
        paddle.nn.AdaptiveAvgPool1D,
        paddle.nn.AdaptiveAvgPool2D,
        paddle.nn.AdaptiveAvgPool3D,
    ),
    "adaptivemax": (
        paddle.nn.AdaptiveMaxPool1D,
        paddle.nn.AdaptiveMaxPool2D,
        paddle.nn.AdaptiveMaxPool3D,
    ),
    "adaptivemaxpool": (
        paddle.nn.AdaptiveMaxPool1D,
        paddle.nn.AdaptiveMaxPool2D,
        paddle.nn.AdaptiveMaxPool3D,
    ),
    "max": (paddle.nn.MaxPool1D, paddle.nn.MaxPool2D, paddle.nn.MaxPool3D),
    "maxpool": (paddle.nn.MaxPool1D, paddle.nn.MaxPool2D, paddle.nn.MaxPool3D),
    "maxunpool": (paddle.nn.MaxUnPool1D, paddle.nn.MaxUnPool2D, paddle.nn.MaxUnPool3D),
    "identity": paddle.nn.Identity,
}


def pooling(
    arg: PoolArg, *args: Any, spatial_dims: Optional[int] = None, **kwargs
) -> paddle.nn.Layer:
    r"""Get pooling layer.

    Args:
        arg: Custom pooling function or module, or name of pooling layer with optional keyword arguments.
            When ``arg`` is a callable but not of type ``paddle.nn.Layer``, it is wrapped in a ``PoolLayer``.
            If ``None`` or 'identity', an instance of ``paddle.nn.Identity`` is returned.
        spatial_dims: Number of spatial dimensions of input tensors.
        args: Arguments to pass to init function of pooling layer. If ``arg`` is a callable, the given arguments
            are passed to the function each time it is called as arguments.
        kwargs: Additional keyword arguments for initialization of pooling layer. Overrides keyword arguments given as
            second tuple item when ``arg`` is a ``(name, kwargs)`` tuple instead of a string. When ``arg`` is a callable,
            the keyword arguments are passed each time the pooling function is called.

    Returns:
        Pooling layer instance.

    """
    if isinstance(arg, paddle.nn.Layer) and not args and not kwargs:
        return arg
    if callable(arg):
        return PoolLayer(arg, *args, **kwargs)
    pool_name = "identity"
    pool_args = {}
    if isinstance(arg, str):
        pool_name = arg
    elif isinstance(arg, Mapping):
        pool_name = arg.get("name")
        if not pool_name:
            raise ValueError("pooling() 'arg' map must contain 'name'")
        if not isinstance(pool_name, str):
            raise TypeError("pooling() 'name' must be str")
        pool_args = {key: value for key, value in arg.items() if key != "name"}
    elif isinstance(arg, Sequence):
        if len(arg) != 2:
            raise ValueError("pooling() 'arg' sequence must have length two")
        pool_name, pool_args = arg
        if not isinstance(pool_name, str):
            raise TypeError("pooling() first 'arg' sequence argument must be str")
        if not isinstance(pool_args, dict):
            raise TypeError("pooling() second 'arg' sequence argument must be dict")
        pool_args = pool_args.copy()
    elif arg is not None:
        raise TypeError("pooling() 'arg' must be str, mapping, 2-tuple, callable, or None")
    pool_type: Union[Type[paddle.nn.Layer], Sequence[Type[paddle.nn.Layer]]] = POOLING_TYPES.get(
        pool_name.lower()
    )
    if pool_type is None:
        raise ValueError(f"pooling() unknown pooling layer {pool_name!r}")
    if isinstance(pool_type, Sequence):
        if spatial_dims is None:
            raise ValueError(f"pooling() 'spatial_dims' required for pooling layer {pool_name!r}")
        try:
            pool_type = pool_type[spatial_dims - 1]
        except IndexError:
            pool_type = None
        if pool_type is None:
            raise ValueError(f"pooling({pool_name!r}) does not support spatial_dims={spatial_dims}")
    pool_args.update(kwargs)
    module = pool_type(*args, **pool_args)
    return module


def pool_layer(*args, **kwargs) -> paddle.nn.Layer:
    return pooling(*args, **kwargs)


class PoolLayer(LambdaLayer):
    r"""Pooling layer."""

    def __init__(
        self, arg: PoolArg, *args: Any, spatial_dims: Optional[int] = None, **kwargs
    ) -> None:
        if callable(arg):
            pool = partial(arg, *args, **kwargs) if args or kwargs else arg
        else:
            pool = pooling(arg, *args, spatial_dims=spatial_dims, **kwargs)
        super().__init__(pool)

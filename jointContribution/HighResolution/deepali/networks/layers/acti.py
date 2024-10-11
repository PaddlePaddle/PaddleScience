from functools import partial
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Type
from typing import Union

import paddle

from .lambd import LambdaLayer

ActivationFunc = Callable[[paddle.Tensor], paddle.Tensor]
ActivationArg = Union[ActivationFunc, str, Mapping, Sequence, None]
ACTIVATION_TYPES = {
    "celu": paddle.nn.CELU,
    "elu": paddle.nn.ELU,
    "hardtanh": paddle.nn.Hardtanh,
    "identity": paddle.nn.Identity,
    "none": paddle.nn.Identity,
    "relu": paddle.nn.ReLU,
    "relu6": paddle.nn.ReLU6,
    "lrelu": paddle.nn.LeakyReLU,
    "leakyrelu": paddle.nn.LeakyReLU,
    "leaky_relu": paddle.nn.LeakyReLU,
    "rrelu": paddle.nn.RReLU,
    "selu": paddle.nn.SELU,
    "gelu": paddle.nn.GELU,
    "hardshrink": paddle.nn.Hardshrink,
    "hardsigmoid": paddle.nn.Hardsigmoid,
    "hardswish": paddle.nn.Hardswish,
    "logsigmoid": paddle.nn.LogSigmoid,
    "logsoftmax": paddle.nn.LogSoftmax,
    "prelu": paddle.nn.PReLU,
    "sigmoid": paddle.nn.Sigmoid,
    "softmax": paddle.nn.Softmax,
    "softmin": paddle.nn.Softmin,
    "softplus": paddle.nn.Softplus,
    "softshrink": paddle.nn.Softshrink,
    "softsign": paddle.nn.Softsign,
    "tanh": paddle.nn.Tanh,
    "tanhshrink": paddle.nn.Tanhshrink,
}
INPLACE_ACTIVATIONS = {
    "elu",
    "hardtanh",
    "lrelu",
    "leakyrelu",
    "relu",
    "relu6",
    "rrelu",
    "selu",
    "celu",
}
SOFTMINMAX_ACTIVATIONS = {"softmin", "softmax", "logsoftmax"}


def activation(
    arg: ActivationArg,
    *args: Any,
    dim: Optional[int] = None,
    inplace: Optional[bool] = None,
    **kwargs,
) -> paddle.nn.Layer:
    """Get activation function.

    Args:
        arg: Custom activation function or module, or name of activation function with optional keyword arguments.
        args: Arguments to pass to activation init function.
        dim: Dimension along which to compute softmax activations (cf. ``ACT_SOFTMINMAX``). Unused by other activations.
        inplace: Whether to compute activation output in place. Unused if unsupported by specified activation function.
        kwargs: Additional keyword arguments for activation function. Overrides keyword arguments given as second
            tuple item when ``arg`` is a ``(name, kwargs)`` tuple instead of a string.

    Returns:
        Given activation function when ``arg`` is a ``paddle.nn.Layer``, or a new activation module otherwise.

    """
    if isinstance(arg, paddle.nn.Layer) and not args and not kwargs:
        return arg
    if callable(arg):
        return Activation(arg, *args, **kwargs)
    acti_name = "identity"
    acti_args = {}
    if isinstance(arg, str):
        acti_name = arg
    elif isinstance(arg, Mapping):
        acti_name = arg.get("name")
        if not acti_name:
            raise ValueError("activation() 'arg' map must contain 'name'")
        if not isinstance(acti_name, str):
            raise TypeError("activation() 'name' must be str")
        acti_args = {key: value for key, value in arg.items() if key != "name"}
    elif isinstance(arg, Sequence):
        if len(arg) != 2:
            raise ValueError("activation() 'arg' sequence must have length two")
        acti_name, acti_args = arg
        if not isinstance(acti_name, str):
            raise TypeError("activation() first 'arg' sequence argument must be str")
        if not isinstance(acti_args, dict):
            raise TypeError("activation() second 'arg' sequence argument must be dict")
        acti_args = acti_args.copy()
    elif arg is not None:
        raise TypeError(
            "activation() 'arg' must be str, mapping, 2-tuple, callable, or None"
        )
    acti_name = acti_name.lower()
    acti_type: Type[paddle.nn.Layer] = ACTIVATION_TYPES.get(acti_name)
    if acti_type is None:
        raise ValueError(
            f"activation() 'arg' name {acti_name!r} is unknown. Pass a callable activation function or module instead."
        )
    acti_args.update(kwargs)
    if inplace is not None and acti_name in INPLACE_ACTIVATIONS:
        acti_args["inplace"] = bool(inplace)
    if acti_name in SOFTMINMAX_ACTIVATIONS:
        if dim is None and len(args) == 1 and isinstance(args, int):
            dim = args[0]
        elif args or acti_args:
            raise ValueError("activation() named {act_name!r} has no parameters")
        if dim is None:
            dim = 1
        acti = acti_type(dim)
    else:
        acti = acti_type(*args, **acti_args)
    return acti


class Activation(LambdaLayer):
    """Non-linear activation function."""

    def __init__(
        self,
        arg: ActivationArg,
        *args: Any,
        dim: int = 1,
        inplace: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if callable(arg):
            acti = partial(arg, *args, **kwargs) if args or kwargs else arg
        else:
            acti = activation(arg, *args, dim=dim, inplace=inplace, **kwargs)
        super().__init__(acti)


def is_activation(arg: Any) -> bool:
    """Whether given object is an non-linear activation function module."""
    if isinstance(arg, Activation):
        return True
    types = tuple(ACTIVATION_TYPES.values())
    return isinstance(arg, types)

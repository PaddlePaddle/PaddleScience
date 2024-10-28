r"""Join features from separate network paths."""

from typing import Callable
from typing import Sequence
from typing import Union

import paddle

from .lambd import LambdaLayer

JoinFunc = Callable[[Sequence[paddle.Tensor]], paddle.Tensor]


def join_func(arg: Union[str, JoinFunc], dim: int = 1) -> JoinFunc:
    r"""Tensor operation which combines features of input tensors, e.g., along skip connection.

    Args:
        arg: Name of operation: 'add': Elementwise addition, 'cat' or 'concat': Concatenate along feature dimension.
        dim: Dimension of input tensors containing features.

    """
    if callable(arg):
        return arg

    if not isinstance(arg, str):
        raise TypeError("join_func() 'arg' must be str or callable")

    name = arg.lower()
    if name == "add":

        def add(args: Sequence[paddle.Tensor]) -> paddle.Tensor:
            assert args, "join_func('add') requires at least one input tensor"
            out = args[0]
            for i in range(1, len(args)):
                out = out + args[i]
            return out

        return add

    elif name in ("cat", "concat"):

        def cat(args: Sequence[paddle.Tensor]) -> paddle.Tensor:
            assert args, "join_func('cat') requires at least one input tensor"
            return paddle.concat(x=args, axis=dim)

        return cat

    elif name == "mul":

        def mul(args: Sequence[paddle.Tensor]) -> paddle.Tensor:
            assert args, "join_func('mul') requires at least one input tensor"
            out = args[0]
            for i in range(1, len(args)):
                out = out * args[i]
            return out

        return mul

    raise ValueError("join_func() unknown merge function {name!r}")


class JoinLayer(LambdaLayer):
    r"""Merge network branches."""

    def __init__(self, arg: Union[str, JoinFunc], dim: int = 1) -> None:
        func = join_func(arg, dim=dim)
        super().__init__(func)

    def forward(self, xs: Sequence[paddle.Tensor]) -> paddle.Tensor:
        return self.func(xs)

    def extra_repr(self) -> str:
        return repr(self.func.__name__)

from typing import Any
from typing import Callable
from typing import Mapping
from typing import Union
from typing import overload

import paddle
import paddle.nn.Layer as Layer

from ...modules.mixins import ReprWithCrossReferences
from ..layers.join import JoinFunc
from ..layers.join import join_func

__all__ = "DenseBlock", "Shortcut", "SkipConnection", "SkipFunc"
SkipFunc = Callable[[paddle.Tensor], paddle.Tensor]


class DenseBlock(ReprWithCrossReferences, paddle.nn.Layer):
    """Subnetwork with dense skip connections."""

    @overload
    def __init__(
        self, *args: paddle.nn.Layer, join: Union[str, JoinFunc], dim: int
    ) -> None:
        ...

    @overload
    def __init__(
        self, arg: Mapping[str, Layer], join: Union[str, JoinFunc], dim: int
    ) -> None:
        ...

    def __init__(
        self, *args: Any, join: Union[str, JoinFunc] = "cat", dim: int = 1
    ) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], Mapping):
            layers = args[0]
        else:
            layers = {str(i): m for i, m in enumerate(args)}
        self.layers = paddle.nn.LayerDict(sublayers=layers)
        self.join = join_func(join, dim=dim)
        self.is_associative = join in ("add", "cat", "concat")

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        y, ys = x, [x]
        join = self.join
        is_associative = self.is_associative
        for module in self.layers.values():
            x = join(ys)
            y = module(x)
            ys = [x, y] if is_associative else [*ys, y]
        return y


class SkipConnection(ReprWithCrossReferences, paddle.nn.Layer):
    """Combine input with subnetwork output along a single skip connection."""

    @overload
    def __init__(
        self,
        *args: paddle.nn.Layer,
        name: str = "func",
        skip: Union[str, SkipFunc] = "identity",
        join: Union[str, JoinFunc] = "cat",
        dim: int = 1
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: Mapping[str, Layer],
        name: str = "func",
        skip: Union[str, SkipFunc] = "identity",
        join: Union[str, JoinFunc] = "cat",
        dim: int = 1,
    ) -> None:
        ...

    def __init__(
        self,
        *args: Any,
        name: str = "func",
        skip: Union[str, SkipFunc] = "identity",
        join: Union[str, JoinFunc] = "cat",
        dim: int = 1
    ) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], paddle.nn.Layer):
            func = args[0]
        else:
            func = paddle.nn.Sequential(*args)
        self.name = name
        if skip in (None, "identity"):
            skip = paddle.nn.Identity()
        elif not callable(skip):
            raise ValueError(
                "SkipConnection() 'skip' must be 'identity', callable, or None"
            )
        self.skip = skip
        self.join = join_func(join, dim=dim)
        self._modules[self.name] = func

    @property
    def func(self) -> paddle.nn.Layer:
        return self._modules[self.name]

    @property
    def shortcut(self) -> SkipFunc:
        return self.skip

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        a = self.skip(x)
        b = self.func(x)
        if not isinstance(a, paddle.Tensor):
            raise TypeError(
                "SkipConnection() 'skip' function must return a paddle.Tensor"
            )
        if not isinstance(b, paddle.Tensor):
            raise TypeError("SkipConnection() module must return a paddle.Tensor")
        c = self.join([b, a])
        if not isinstance(c, paddle.Tensor):
            raise TypeError(
                "SkipConnection() 'join' function must return a paddle.Tensor"
            )
        return c


Shortcut = SkipConnection

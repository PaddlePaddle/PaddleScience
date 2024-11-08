from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import paddle

from ..core import functional as U
from ..core.enum import PaddingMode
from ..core.types import ScalarOrTuple


class GetItem(paddle.nn.Layer):
    """Get item at specified input tensor sequence index or with given dictionary key."""

    def __init__(self, key: Any) -> None:
        """Set item index or key.

        Args:
            key: Index of item in sequence of input tensors or key into input map.

        """
        super().__init__()
        self.key = key

    def forward(
        self, input: Union[Sequence[paddle.Tensor], Mapping[Any, paddle.Tensor]]
    ) -> paddle.Tensor:
        return input[self.key]

    def extra_repr(self) -> str:
        return repr(self.key)


class Narrow(paddle.nn.Layer):
    """Narrowed version of input tensor."""

    def __init__(self, dim: int, start: int, length: int) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        start_6 = x.shape[self.dim] + self.start if self.start < 0 else self.start
        return paddle.slice(x, [self.dim], [start_6], [start_6 + self.length])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, start={self.start}, length={self.length}"


class Pad(paddle.nn.Layer):
    """Pad tensor."""

    def __init__(
        self,
        margin: Optional[ScalarOrTuple[int]] = None,
        padding: Optional[ScalarOrTuple[int]] = None,
        mode: Union[PaddingMode, str] = PaddingMode.ZEROS,
        value: float = 0,
    ) -> None:
        if margin is None and padding is None:
            raise AssertionError("Pad() either 'margin' or 'padding' is required")
        if margin is not None and padding is not None:
            raise AssertionError("Pad() 'margin' and 'padding' are mutually exclusive")
        super().__init__()
        self.margin = margin
        self.padding = padding
        self.mode = PaddingMode(mode)
        self.value = value

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return U.pad(
            x, margin=self.margin, num=self.padding, mode=self.mode, value=self.value
        )

    def extra_repr(self) -> str:
        if self.margin is None:
            s = f"padding={self.padding}"
        else:
            s = f"margin={self.margin}"
        s += f", mode={self.mode.value!r}"
        if self.mode is PaddingMode.CONSTANT:
            s += f", value={self.value}"
        return s


class Reshape(paddle.nn.Layer):
    """Reshape input tensor.

    This module provides a view of the input tensor without making a copy if possible.
    Otherwise, a copy is made of the input data. See ``paddle.reshape()`` for details.

    """

    def __init__(self, shape: Sequence[int]) -> None:
        """Set output tensor shape.

        Args:
            shape: Output tensor shape, optionally excluding first batch dimension.

        """
        super().__init__()
        self.shape = shape

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shape = self.shape
        if len(shape) == x.ndim - 1:
            shape = (-1,) + shape
        return x.reshape(shape)

    def extra_repr(self) -> str:
        return repr(self.shape)


class View(paddle.nn.Layer):
    """View input tensor with specified shape."""

    def __init__(self, shape: Sequence[int]) -> None:
        """Set output tensor shape.

        Args:
            shape: Output tensor shape, optionally excluding first batch dimension.

        """
        super().__init__()
        self.shape = shape

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shape = self.shape
        if len(shape) == x.ndim - 1:
            shape = (-1,) + shape
        return x.view(*shape)

    def extra_repr(self) -> str:
        return repr(self.shape)

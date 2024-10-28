r"""Modules operating on flow vector fields."""

from __future__ import annotations  # noqa

from copy import copy as shallow_copy
from typing import Optional
from typing import Union

import paddle
from deepali.core import ALIGN_CORNERS
from deepali.core import Array
from deepali.core import Scalar
from deepali.core import ScalarOrTuple
from deepali.core import functional as U


class Curl(paddle.nn.Layer):
    r"""Layer which calculates the curl of a vector field."""

    def __init__(
        self,
        mode: Optional[str] = None,
        sigma: Optional[float] = None,
        spacing: Optional[Union[Scalar, Array]] = None,
        stride: Optional[ScalarOrTuple[int]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.sigma = sigma
        self.spacing = spacing
        self.stride = stride

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return U.curl(x, mode=self.mode, sigma=self.sigma, spacing=self.spacing, stride=self.stride)

    def extra_repr(self) -> str:
        args = []
        if self.mode is not None:
            args.append(f"mode={self.mode!r}")
        if self.sigma is not None:
            args.append(f"sigma={self.sigma!r}")
        if self.spacing is not None:
            args.append(f"spacing={self.spacing!r}")
        if self.stride is not None:
            args.append(f"stride={self.stride!r}")
        return ", ".join(args)


class ExpFlow(paddle.nn.Layer):
    r"""Layer that computes exponential map of flow field."""

    def __init__(
        self,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
        align_corners: bool = ALIGN_CORNERS,
    ):
        r"""Initialize parameters.

        Args:
            scale: Constant scaling factor of input velocities (e.g., -1 for inverse). Default is 1.
            steps: Number of squaring steps.
            align_corners: Whether input vectors are with respect to ``Axes.CUBE`` (False)
                or ``Axes.CUBE_CORNERS`` (True). This flag is passed on to ``grid_sample()``.

        """
        super().__init__()
        self.scale = float(1 if scale is None else scale)
        self.steps = int(5 if steps is None else steps)
        self.align_corners = bool(align_corners)

    def forward(self, x: paddle.Tensor, inverse: bool = False) -> paddle.Tensor:
        r"""Compute exponential map of vector field."""
        scale = self.scale
        if inverse:
            scale *= -1
        return U.expv(x, scale=scale, steps=self.steps, align_corners=self.align_corners)

    @property
    def inv(self) -> ExpFlow:
        r"""Get inverse exponential map.

        .. code-block:: python

            u = exp(v)
            w = exp.inv(v)

        """
        return self.inverse()

    def inverse(self) -> ExpFlow:
        r"""Get inverse exponential map."""
        copy = shallow_copy(self)
        copy.scale *= -1
        return copy

    def extra_repr(self) -> str:
        return f"scale={repr(self.scale)}, steps={repr(self.steps)}"

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from __future__ import annotations

import math

import numpy as np
import paddle

from ppsci.arch.aurora.area import radius_earth

__all__ = [
    "FourierExpansion",
    "pos_expansion",
    "scale_expansion",
    "lead_time_expansion",
    "lead_time_expansion_v3",
    "levels_expansion",
    "absolute_time_expansion",
]


class FourierExpansion(paddle.nn.Layer):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: paddle.Tensor, d: int) -> paddle.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`paddle.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            paddle.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        in_range = paddle.logical_and(
            self.lower <= x.abs(), paddle.all(x.abs() <= self.upper)
        )
        in_range_or_zero = paddle.all(paddle.logical_or(in_range, x == 0))
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range `[{self.lower}, {self.upper}]`."
            )
        if not d % 2 == 0:
            raise ValueError("The dimensionality must be a multiple of two.")
        x = x.double()
        wavelengths = paddle.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            dtype=x.dtype,
        ).to(device=x.device)
        prod = paddle.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = paddle.cat((paddle.sin(prod), paddle.cos(prod)), axis=-1)
        return encoding.float()


class FourierExpansionWithLinear(FourierExpansion):
    """Fourier expansion that replaces the sin of the largest wavelength with a linear feature.

    Since `sin(2 pi x / lambda) ~ 2 pi x / lambda` for `x << lambda`, the sin channel at the
    largest wavelength carries almost the same information as a simple linear term. Replacing it
    with `x / upper` gives the model a clean linear signal while keeping the total dimensionality
    unchanged so that checkpoints remain compatible.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def forward(self, x: paddle.Tensor, d: int) -> paddle.Tensor:
        """Perform the expansion, replacing the last sin channel with a linear feature.

        Args:
            x (:class:`paddle.Tensor`): Input to expand of shape `(..., n)`.
            d (int): Dimensionality.  Must be a multiple of two.

        Returns:
            paddle.Tensor: Expansion of `x` of shape `(..., n, d)` where the channel at index
                `d // 2 - 1` (sin of the largest wavelength) is replaced by
                `x / self.upper`.
        """
        encoding = super().forward(x, d)
        linear = (x / self.upper).float().unsqueeze(-1)
        encoding = paddle.cat(
            [encoding[..., : d // 2 - 1], linear, encoding[..., d // 2 :]], axis=-1
        )
        return encoding


_delta = 0.01
# Preserve the Torch implementation's CPU float64 calibration constant. The
# near-pole area calculation suffers cancellation: an O(1e-8) relative change
# alters the phase of the highest-frequency spatial features significantly.
_min_patch_area: float = 0.00010814085263058073
_area_earth = 4 * np.pi * radius_earth * radius_earth
pos_expansion = FourierExpansion(_delta, 720)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of latitudes and longitudes in
degrees."""
scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of patch areas in squared
kilometers."""
lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)
""":class:`.FourierExpansion`: Fourier expansion for the lead time encoding in hours."""
lead_time_expansion_v3 = FourierExpansionWithLinear(6, 24 * 7 * 3, assert_range=False)
""":class:`.FourierExpansionWithLinear`: Updated Fourier expansion for the lead time encoding in
hours. Uses a minimum lead time of 6 hours to avoid highly-varying waves at short lead times and
replaces the sin of the largest wavelength with a normalised linear feature `x / upper`."""
levels_expansion = FourierExpansion(0.01, 100000.0)
""":class:`.FourierExpansion`: Fourier expansion for the pressure level encoding in hPa."""
absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
""":class:`.FourierExpansion`: Fourier expansion for the absolute time encoding in hours."""

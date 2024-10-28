r"""Flow field regularization terms."""
from __future__ import annotations  # noqa

from typing import Optional
from typing import Union

import paddle
from deepali.core.typing import Array
from deepali.core.typing import Scalar
from deepali.core.typing import ScalarOrTuple

from . import functional as L
from .base import DisplacementLoss


class _SpatialDerivativesLoss(DisplacementLoss):
    r"""Base class of regularization terms based on spatial derivatives of dense displacements."""

    def __init__(
        self,
        mode: Optional[str] = None,
        sigma: Optional[float] = None,
        spacing: Optional[Union[Scalar, Array]] = None,
        stride: Optional[ScalarOrTuple] = None,
        reduction: str = "mean",
    ):
        r"""Initialize regularization term.

        Args:
            mode: Method used to approximate :func:`flow_derivatives()`.
            sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
            spacing: Spacing between grid elements. Should be given in the units of the flow vectors.
                By default, flow vectors with respect to normalized grid coordinates are assumed.
            stride: Number of output grid points between control points plus one for ``mode='bspline'``.
            reduction: Operation to use for reducing spatially distributed loss values.

        """
        super().__init__()
        self.mode = mode
        self.sigma = sigma
        self.spacing = spacing
        self.stride = stride
        self.reduction = reduction

    def extra_repr(self) -> str:
        args = []
        if self.mode:
            args.append(f"mode={self.mode!r}")
        if self.sigma:
            args.append(f"sigma={self.sigma!r}")
        if self.spacing:
            args.append(f"spacing={self.spacing!r}")
        if self.stride:
            args.append(f"stride={self.stride!r}")
        args.append(f"reduction={self.reduction!r}")
        return ", ".join(args)


class GradLoss(_SpatialDerivativesLoss):
    r"""Displacement field gradient loss."""

    def __init__(
        self,
        p: Union[int, float] = 2,
        q: Optional[Union[int, float]] = 1,
        mode: Optional[str] = None,
        sigma: Optional[float] = None,
        spacing: Optional[Union[Scalar, Array]] = None,
        stride: Optional[ScalarOrTuple] = None,
        reduction: str = "mean",
    ):
        r"""Initialize regularization term.

        Args:
            mode: Method used to approximate :func:`flow_derivatives()`.
            sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
            spacing: Spacing between grid elements. Should be given in the units of the flow vectors.
                By default, flow vectors with respect to normalized grid coordinates are assumed.
            stride: Number of output grid points between control points plus one for ``mode='bspline'``.
            reduction: Operation to use for reducing spatially distributed loss values.

        """
        super().__init__(
            mode=mode, sigma=sigma, spacing=spacing, stride=stride, reduction=reduction
        )
        self.p = p
        self.q = 1 / p if q is None else q

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.grad_loss(
            u,
            p=self.p,
            q=self.q,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"p={self.p}, q={self.q}, " + super().extra_repr()


class Bending(_SpatialDerivativesLoss):
    r"""Bending energy of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.bending_loss(
            u,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )


BendingEnergy = Bending
BE = Bending


class Curvature(_SpatialDerivativesLoss):
    r"""Curvature of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.curvature_loss(
            u,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )


class Diffusion(_SpatialDerivativesLoss):
    r"""Diffusion of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.diffusion_loss(
            u,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )


class Divergence(_SpatialDerivativesLoss):
    r"""Divergence of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.divergence_loss(
            u,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )


class Elasticity(_SpatialDerivativesLoss):
    r"""Linear elasticity of displacement field."""

    def __init__(
        self,
        material_name: Optional[str] = None,
        first_parameter: Optional[float] = None,
        second_parameter: Optional[float] = None,
        poissons_ratio: Optional[float] = None,
        youngs_modulus: Optional[float] = None,
        shear_modulus: Optional[float] = None,
        mode: Optional[str] = None,
        sigma: Optional[float] = None,
        spacing: Optional[Union[Scalar, Array]] = None,
        stride: Optional[ScalarOrTuple] = None,
        reduction: str = "mean",
    ):
        super().__init__(mode=mode, sigma=sigma, spacing=spacing, reduction=reduction)
        self.material_name = material_name
        self.first_parameter = first_parameter
        self.second_parameter = second_parameter
        self.poissons_ratio = poissons_ratio
        self.youngs_modulus = youngs_modulus
        self.shear_modulus = shear_modulus

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.elasticity_loss(
            u,
            material_name=self.material_name,
            first_parameter=self.first_parameter,
            second_parameter=self.second_parameter,
            poissons_ratio=self.poissons_ratio,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        args = []
        if self.material_name:
            args.append(f"material_name={self.material_name!r}")
        if self.first_parameter is not None:
            args.append(f"first_parameter={self.first_parameter!r}")
        if self.second_parameter is not None:
            args.append(f"second_parameter={self.second_parameter!r}")
        if self.poissons_ratio is not None:
            args.append(f"poissons_ratio={self.poissons_ratio!r}")
        if self.youngs_modulus is not None:
            args.append(f"youngs_modulus={self.youngs_modulus!r}")
        if self.shear_modulus is not None:
            args.append(f"shear_modulus={self.shear_modulus!r}")
        return ", ".join(args) + ", " + super().extra_repr()


class TotalVariation(_SpatialDerivativesLoss):
    r"""Total variation of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        return L.total_variation_loss(
            u,
            mode=self.mode,
            sigma=self.sigma,
            spacing=self.spacing,
            stride=self.stride,
            reduction=self.reduction,
        )


TV = TotalVariation

from __future__ import annotations

from typing import Optional
from typing import Union

import paddle

from ..core.types import Shape
from . import functional as L
from .base import DisplacementLoss


class _SpatialDerivativesLoss(DisplacementLoss):
    """Base class of regularization terms based on spatial derivatives of dense displacements."""

    def __init__(
        self,
        mode: str = "central",
        sigma: Optional[float] = None,
        reduction: str = "mean",
    ):
        """Initialize regularization term.

        Args:
            mode: Method used to approximate spatial derivatives. See ``spatial_derivatives()``.
            sigma: Standard deviation of Gaussian in grid units. See ``spatial_derivatives()``.
            reduction: Operation to use for reducing spatially distributed loss values.

        """
        super().__init__()
        self.mode = mode
        self.sigma = float(0 if sigma is None else sigma)
        self.reduction = reduction

    def _spacing(self, u_shape: Shape) -> Optional[paddle.Tensor]:
        ndim = len(u_shape)
        if ndim < 3:
            raise ValueError(
                f"{type(self).__name__}.forward() 'u' must be at least 3-dimensional"
            )
        if ndim == 3:
            return None
        size = paddle.to_tensor(
            data=u_shape[-1:1:-1],
            dtype="float32",
            place=str("cpu").replace("cuda", "gpu"),
        )
        return 2 / (size - 1)

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, sigma={self.sigma!r}, reduction={self.reduction!r}"


class GradLoss(_SpatialDerivativesLoss):
    """Displacement field gradient loss."""

    def __init__(
        self,
        p: Union[int, float] = 2,
        q: Optional[Union[int, float]] = 1,
        mode: str = "central",
        sigma: Optional[float] = None,
        reduction: str = "mean",
    ):
        """Initialize regularization term.

        Args:
            mode: Method used to approximate spatial derivatives. See ``spatial_derivatives()``.
            sigma: Standard deviation of Gaussian in grid units. See ``spatial_derivatives()``.
            reduction: Operation to use for reducing spatially distributed loss values.

        """
        super().__init__(mode=mode, sigma=sigma, reduction=reduction)
        self.p = p
        self.q = q

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.grad_loss(
            u,
            p=self.p,
            q=self.q,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"p={self.p}, q={self.q}, " + super().extra_repr()


class Bending(_SpatialDerivativesLoss):
    """Bending energy of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.bending_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


BendingEnergy = Bending
BE = Bending


class Curvature(_SpatialDerivativesLoss):
    """Curvature of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.curvature_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class Diffusion(_SpatialDerivativesLoss):
    """Diffusion of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.diffusion_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class Divergence(_SpatialDerivativesLoss):
    """Divergence of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.divergence_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


class Elasticity(_SpatialDerivativesLoss):
    """Linear elasticity of displacement field."""

    def __init__(
        self,
        material_name: Optional[str] = None,
        first_parameter: Optional[float] = None,
        second_parameter: Optional[float] = None,
        poissons_ratio: Optional[float] = None,
        youngs_modulus: Optional[float] = None,
        shear_modulus: Optional[float] = None,
        mode: str = "central",
        sigma: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__(mode=mode, sigma=sigma, reduction=reduction)
        self.material_name = material_name
        self.first_parameter = first_parameter
        self.second_parameter = second_parameter
        self.poissons_ratio = poissons_ratio
        self.youngs_modulus = youngs_modulus
        self.shear_modulus = shear_modulus

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.elasticity_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
            material_name=self.material_name,
            first_parameter=self.first_parameter,
            second_parameter=self.second_parameter,
            poissons_ratio=self.poissons_ratio,
            youngs_modulus=self.youngs_modulus,
            shear_modulus=self.shear_modulus,
        )


class TotalVariation(_SpatialDerivativesLoss):
    """Total variation of displacement field."""

    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        """Evaluate regularization loss for given transformation."""
        spacing = self._spacing(tuple(u.shape))
        return L.total_variation_loss(
            u,
            spacing=spacing,
            mode=self.mode,
            sigma=self.sigma,
            reduction=self.reduction,
        )


TV = TotalVariation

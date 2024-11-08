import paddle

from . import functional as L
from .base import BSplineLoss


class BSplineBending(BSplineLoss):
    """Bending energy of cubic B-spline free form deformation."""

    def forward(self, params: paddle.Tensor) -> paddle.Tensor:
        """Evaluate loss term for given free form deformation parameters."""
        return L.bspline_bending_loss(
            params, stride=self.stride, reduction=self.reduction
        )


BSplineBendingEnergy = BSplineBending

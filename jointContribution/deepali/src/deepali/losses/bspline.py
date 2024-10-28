r"""Free-form deformation (FFD) regularization terms."""
import paddle

from . import functional as L
from .base import BSplineLoss


class BSplineBending(BSplineLoss):
    r"""Bending energy of cubic B-spline free form deformation."""

    def forward(self, params: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate loss term for given free form deformation parameters."""
        return L.bending_loss(params, mode="bspline", stride=self.stride, reduction=self.reduction)


BSplineBendingEnergy = BSplineBending

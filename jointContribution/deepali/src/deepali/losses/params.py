r"""Regularization terms based on model parameters."""
import paddle

from .base import ParamsLoss


class L1Norm(ParamsLoss):
    r"""Regularization loss term based on L1-norm of model parameters."""

    def __init__(self, scale: float = 1000.0) -> None:
        r"""Initialize loss term.

        Args:
            scale: Constant factor by which to scale loss value such that magnitude is in
                similar range to other registration loss terms, i.e., image matching terms.

        """
        super().__init__()
        self.scale = scale

    def forward(self, params: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate loss term for given model parameters."""
        return self.scale * params.abs().mean()


L1_Norm = L1Norm


class L2Norm(ParamsLoss):
    r"""Regularization loss term based on L2-norm of model parameters."""

    def __init__(self, scale: float = 1000000.0) -> None:
        r"""Initialize loss term.

        Args:
            scale: Constant factor by which to scale loss value such that magnitude is in
                similar range to other registration loss terms, i.e., image matching terms.

        """
        super().__init__()
        self.scale = scale

    def forward(self, params: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate loss term for given model parameters."""
        return self.scale * params.square().mean()


L2_Norm = L2Norm


class Sparsity(ParamsLoss):
    r"""Regularization loss term encouraging sparsity of non-zero model parameters."""

    def forward(self, params: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate loss term for given model parameters."""
        return params.abs().mean()

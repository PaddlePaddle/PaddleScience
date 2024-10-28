r"""Image dissimilarity measures."""
from typing import Optional
from typing import Union

import deepali.utils.paddle_aux  # noqa
import paddle
from deepali.core import functional as U
from deepali.core.typing import ScalarOrTuple

from . import functional as L
from .base import NormalizedPairwiseImageLoss
from .base import PairwiseImageLoss


class Dice(PairwiseImageLoss):
    r"""Generalized Sorensen-Dice similarity coefficient."""

    def __init__(self, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.dice_loss(source, target, weight=mask, epsilon=self.epsilon)

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon:.2e}"


DSC = Dice


class NCC(PairwiseImageLoss):
    r"""Normalized cross correlation."""

    def __init__(self, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.ncc_loss(source, target, mask=mask, epsilon=self.epsilon)

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon:.2e}"


class LCC(PairwiseImageLoss):
    r"""Local normalized cross correlation."""

    def __init__(self, kernel_size: ScalarOrTuple[int] = 7, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.lcc_loss(
            source, target, mask=mask, kernel_size=self.kernel_size, epsilon=self.epsilon
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, epsilon={self.epsilon:.2e}"


LNCC = LCC


class WLCC(PairwiseImageLoss):
    r"""Weighted local normalized cross correlation."""

    def __init__(self, kernel_size: ScalarOrTuple[int] = 7, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def forward(
        self,
        source: paddle.Tensor,
        target: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None,
        source_mask: Optional[paddle.Tensor] = None,
        target_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.wlcc_loss(
            source,
            target,
            mask=mask,
            source_mask=source_mask,
            target_mask=target_mask,
            kernel_size=self.kernel_size,
            epsilon=self.epsilon,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, epsilon={self.epsilon:.2e}"


SLCC = WLCC  # Sparse local cross correlation (cf. Lewis et al. 2020, arXiv:1812.06932)


class L1ImageLoss(NormalizedPairwiseImageLoss):
    r"""Average absolute intensity differences."""

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.mae_loss(source, target, mask=mask, norm=self.norm)


MAE = L1ImageLoss


class HuberImageLoss(NormalizedPairwiseImageLoss):
    r"""Average Huber loss."""

    def __init__(
        self,
        source: Optional[paddle.Tensor] = None,
        target: Optional[paddle.Tensor] = None,
        norm: Optional[Union[bool, paddle.Tensor]] = None,
        delta: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        r"""Initialize similarity metric.

        Args:
            source: Source image from which to compute ``norm``. If ``None``, only use ``target`` if specified.
            target: Target image from which to compute ``norm``. If ``None``, only use ``source`` if specified.
            norm: Positive factor by which to divide loss. If ``None`` or ``True``, use ``source`` and/or ``target``.
                If ``False`` or both ``source`` and ``target`` are ``None``, a normalization factor of one is used.
            delta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
            beta: Alternative name for ``delta`` to be compatible with ``SmoothL1ImageLoss``.

        """
        if beta is not None:
            if delta is not None:
                raise ValueError(
                    f"{type(self).__name__}() 'delta' and 'beta' are mutually exclusive"
                )
            delta = beta
        elif delta is None:
            delta = 1.0
        super().__init__(source, target, norm)
        self.delta = delta

    @property
    def beta(self) -> float:
        return self.delta

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.huber_loss(source, target, mask=mask, norm=self.norm, delta=self.delta)


class SmoothL1ImageLoss(NormalizedPairwiseImageLoss):
    r"""Average smooth L1 loss."""

    def __init__(
        self,
        source: Optional[paddle.Tensor] = None,
        target: Optional[paddle.Tensor] = None,
        norm: Optional[Union[bool, paddle.Tensor]] = None,
        beta: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        r"""Initialize similarity metric.

        Args:
            source: Source image from which to compute ``norm``. If ``None``, only use ``target`` if specified.
            target: Target image from which to compute ``norm``. If ``None``, only use ``source`` if specified.
            norm: Positive factor by which to divide loss. If ``None`` or ``True``, use ``source`` and/or ``target``.
                If ``False`` or both ``source`` and ``target`` are ``None``, a normalization factor of one is used.
            beta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.
            delta: Alternative name for ``beta`` to be compatible with ``HuberImageLoss``.

        """
        if delta is not None:
            if beta is not None:
                raise ValueError(
                    f"{type(self).__name__}() 'beta' and 'detla' are mutually exclusive"
                )
            beta = delta
        elif beta is None:
            beta = 1.0
        super().__init__(source, target, norm)
        self.beta = beta

    @property
    def delta(self) -> float:
        return self.beta

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.smooth_l1_loss(source, target, mask=mask, norm=self.norm)


class L2ImageLoss(NormalizedPairwiseImageLoss):
    r"""Average squared intensity differences."""

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.mse_loss(source, target, mask=mask, norm=self.norm)


MSE = L2ImageLoss


class SSD(NormalizedPairwiseImageLoss):
    r"""Sum of squared intensity differences."""

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        return L.ssd_loss(source, target, mask=mask, norm=self.norm)


class MI(PairwiseImageLoss):
    r"""Mutual information loss using Parzen window estimate with Gaussian kernel."""

    def __init__(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        bins: Optional[int] = None,
        sample: Optional[float] = None,
        num_bins: Optional[int] = None,
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        normalized: bool = False,
    ):
        r"""Initialize mutual information loss term.

        See :func:`deepali.losses.functional.mi_loss`.

        """

        if bins is not None:
            if num_bins is not None:
                raise ValueError(
                    f"{type(self).__name__}() 'bins' and 'num_bins' are mutually exclusive"
                )
            num_bins = bins

        if sample is not None:
            if sample_ratio is not None or num_samples is not None:
                raise ValueError(
                    f"{type(self).__name__}() 'sample', 'sample_ratio', and 'num_samples' are mutually exclusive"
                )
            if 0 < sample < 1:
                sample_ratio = float(sample)
            else:
                try:
                    num_samples = int(sample)
                except TypeError:
                    pass
                if num_samples is None or float(num_samples) != sample:
                    raise ValueError(
                        f"{type(self).__name__}() 'sample' must be float in (0, 1) or positive int"
                    )
        if num_samples == -1:
            num_samples = None
        if num_samples is not None and (not isinstance(num_samples, int) or num_samples <= 0):
            raise ValueError(
                f"{type(self).__name__}() 'num_samples' must be positive integral value"
            )
        if sample_ratio is not None and (sample_ratio <= 0 or sample_ratio > 1):
            raise ValueError(
                f"{type(self).__name__}() 'sample_ratio' must be in open-closed interval (0, 1]"
            )

        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.sample_ratio = sample_ratio
        self._normalized = normalized

    @property
    def bins(self) -> int:
        return self.num_bins

    @property
    def normalized(self) -> bool:
        return self._normalized

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate patch dissimilarity loss."""
        return L.mi_loss(
            source,
            target,
            mask=mask,
            vmin=self.vmin,
            vmax=self.vmax,
            num_bins=self.num_bins,
            num_samples=self.num_samples,
            sample_ratio=self.sample_ratio,
            normalized=self.normalized,
        )

    def extra_repr(self) -> str:
        return f"vmin={self.vmin!r}, vmax={self.vmax!r}, num_bins={self.num_bins!r}, num_samples={self.num_samples!r}, sampling_ratio={self.sample_ratio!r}, normalized={self.normalized!r}"


class NMI(MI):
    r"""Normalized mutual information loss using Parzen window estimate with Gaussian kernel."""

    def __init__(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        bins: Optional[int] = None,
        sample: Optional[float] = None,
        num_bins: Optional[int] = None,
        num_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
    ):
        r"""Initialize normalized mutual information loss term.

        See :func:`deepali.losses.functional.nmi_loss`.

        """
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            bins=bins,
            sample=sample,
            num_bins=num_bins,
            num_samples=num_samples,
            sample_ratio=sample_ratio,
        )

    def extra_repr(self) -> str:
        return f"vmin={self.vmin!r}, vmax={self.vmax!r}, num_bins={self.num_bins!r}, num_samples={self.num_samples!r}, sampling_ratio={self.sample_ratio!r}"


class PatchwiseImageLoss(PairwiseImageLoss):
    r"""Pairwise similarity of 2D image patches defined within a 3D volume."""

    def __init__(self, patches: paddle.Tensor, loss_fn: PairwiseImageLoss = SSD()):
        r"""Initialize loss term.

        Args:
            patches: Patch sampling points as tensor of shape ``(N, Z, Y, X, 3)``.
            loss_fn: Pairwise image similarity loss term used to evaluate similarity of patches.

        """
        super().__init__()
        if not isinstance(patches, paddle.Tensor):
            raise TypeError("PatchwiseImageLoss() 'patches' must be Tensor")
        if not patches.ndim == 5 or tuple(patches.shape)[-1] != 3:
            raise ValueError("PatchwiseImageLoss() 'patches' must have shape (N, Z, Y, X, 3)")
        self.patches = patches
        self.loss_fn = loss_fn

    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate patch dissimilarity loss."""
        if target.ndim != 5:
            raise ValueError(
                "PatchwiseImageLoss.forward() 'target' must have shape (N, C, Z, Y, X)"
            )
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                "PatchwiseImageLoss.forward() 'source' must have same shape as 'target'"
            )
        if mask is not None:
            if tuple(mask.shape) != tuple(target.shape):
                raise ValueError(
                    "PatchwiseImageLoss.forward() 'mask' must have same shape as 'target'"
                )
            mask = self._reshape(U.grid_sample_mask(mask, self.patches))
        source = self._reshape(U.grid_sample(source, self.patches))
        target = self._reshape(U.grid_sample(target, self.patches))
        return self.loss_fn(source, target, mask=mask)

    @staticmethod
    def _reshape(x: paddle.Tensor) -> paddle.Tensor:
        r"""Reshape tensor to (N * Z, C, 1, Y, X) such that each patch is a separate image in the batch."""
        N, C, Z, Y, X = tuple(x.shape)
        x = x.transpose(perm=[0, 2, 1, 3, 4])  # N, Z, C, Y, X
        x = x.reshape(N * Z, C, 1, Y, X)
        return x


PatchLoss = PatchwiseImageLoss

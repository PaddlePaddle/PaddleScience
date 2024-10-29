r"""Abstract base classes of different loss terms."""
from abc import ABCMeta
from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import paddle
from deepali.core.math import max_difference
from deepali.core.typing import ScalarOrTuple
from paddle import Tensor

RegistrationResult = Dict[str, Any]
RegistrationLosses = Union[
    paddle.nn.Layer,
    paddle.nn.LayerDict,
    paddle.nn.LayerList,
    Mapping[str, paddle.nn.Layer],
    Sequence[paddle.nn.Layer],
]


class RegistrationLoss(paddle.nn.Layer, metaclass=ABCMeta):
    r"""Base class of registration loss functions.

    A registration loss function, also referred to as energy function, is an objective function
    to be minimized by an optimization routine. In particular, these energy functions are used inside
    the main loop which performs individual gradient steps using an instance of ``paddle.optimizer.Optimizer``.

    Registration loss consists of one or more loss terms, which may be either one of:
    - A pairwise data term measuring the alignment of a single input pair (e.g., images, point sets, surfaces).
    - A groupwise data term measuring the alignment of two or more inputs.
    - A regularization term penalizing certain dense spatial deformations.
    - Other regularization terms based on spatial transformation parameters.

    """

    @staticmethod
    def as_module_dict(arg: Optional[RegistrationLosses], start: int = 0) -> paddle.nn.LayerDict:
        r"""Convert argument to ``ModuleDict``."""
        if arg is None:
            return paddle.nn.LayerDict()
        if isinstance(arg, paddle.nn.LayerDict):
            return arg
        if isinstance(arg, paddle.nn.Layer):
            arg = [arg]
        if isinstance(arg, dict):
            modules = arg
        else:
            modules = OrderedDict(
                [
                    (f"loss_{i + start}", m)
                    for i, m in enumerate(arg)
                    if isinstance(m, paddle.nn.Layer)
                ]
            )
        return paddle.nn.LayerDict(sublayers=modules)

    @abstractmethod
    def eval(self) -> RegistrationResult:
        r"""Evaluate registration loss.

        Returns:
            Dictionary of current registration result. The entries in the dictionary depend on the
            respective registration loss function used, but must include at a minimum the total
            scalar "loss" value.

        """
        raise NotImplementedError(f"{type(self).__name__}.eval()")

    def forward(self) -> paddle.Tensor:
        r"""Evaluate registration loss."""
        result = self.eval()
        if not isinstance(result, dict):
            raise TypeError(f"{type(self).__name__}.eval() must return a dictionary")
        if "loss" not in result:
            raise ValueError(f"{type(self).__name__}.eval() result must contain key 'loss'")
        loss = result["loss"]
        if not isinstance(loss, paddle.Tensor):
            raise TypeError(f"{type(self).__name__}.eval() result 'loss' must be tensor")
        return loss


class PairwiseImageLoss(paddle.nn.Layer, metaclass=ABCMeta):
    r"""Base class of pairwise image dissimilarity criteria."""

    @abstractmethod
    def forward(
        self, source: paddle.Tensor, target: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        r"""Evaluate image dissimilarity loss."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class NormalizedPairwiseImageLoss(PairwiseImageLoss):
    r"""Base class of pairwise image dissimilarity criteria with implicit input normalization."""

    def __init__(
        self,
        source: Optional[paddle.Tensor] = None,
        target: Optional[paddle.Tensor] = None,
        norm: Optional[Union[bool, paddle.Tensor]] = None,
    ):
        r"""Initialize similarity metric.

        Args:
            source: Source image from which to compute ``norm``. If ``None``, only use ``target`` if specified.
            target: Target image from which to compute ``norm``. If ``None``, only use ``source`` if specified.
            norm: Positive factor by which to divide loss. If ``None`` or ``True``, use ``source`` and/or ``target``.
                If ``False`` or both ``source`` and ``target`` are ``None``, a normalization factor of one is used.

        """
        super().__init__()
        if norm is True:
            norm = None
        if norm is None:
            if target is None:
                target = source
            elif source is None:
                source = target
            if source is not None and target is not None:
                norm = max_difference(source, target).square()
        elif norm is False:
            norm = None
        assert norm is None or isinstance(norm, (float, int, Tensor))
        self.norm = norm

    def extra_repr(self) -> str:
        s = ""
        norm = self.norm
        if isinstance(norm, paddle.Tensor) and norm.size != 1:
            s += f"norm={self.norm!r}"
        elif norm is not None:
            s += f"norm={float(norm):.5f}"
        return s


class DisplacementLoss(paddle.nn.Layer, metaclass=ABCMeta):
    r"""Base class of regularization terms based on dense displacements."""

    @abstractmethod
    def forward(self, u: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate regularization loss for given transformation."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class BSplineLoss(paddle.nn.Layer, metaclass=ABCMeta):
    r"""Base class of loss terms based on cubic B-spline deformation coefficients."""

    def __init__(self, stride: ScalarOrTuple[int] = 1, reduction: str = "mean"):
        r"""Initialize regularization term.

        Args:
            stride: Number of points between control points at which to evaluate bending energy, plus one.
                If a sequence of values is given, these must be the strides for the different spatial
                dimensions in the order ``(sx, ...)``. A stride of 1 is equivalent to evaluating bending
                energy only at the usually coarser resolution of the control point grid. It should be noted
                that the stride need not match the stride used to densely sample the spline deformation field
                at a given fixed target image resolution.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

        """
        super().__init__()
        self.stride = stride
        self.reduction = reduction

    @abstractmethod
    def forward(self, params: paddle.Tensor, stride: ScalarOrTuple[int] = 1) -> paddle.Tensor:
        r"""Evaluate loss term for given free-form deformation parameters."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")

    def extra_repr(self) -> str:
        return f"stride={self.stride!r}, reduction={self.reduction!r}"


class PointSetDistance(paddle.nn.Layer, metaclass=ABCMeta):
    r"""Base class of point set distance terms."""

    @abstractmethod
    def forward(self, x: paddle.Tensor, *ys: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate point set distance.

        Note that some point set distance measures require a 1-to-1 correspondence
        between the two input point sets, and thus ``M == N``. Other distance losses
        may compute correspondences themselves, e.g., based on closest points.

        Args:
            x: Tensor of shape ``(M, X, D)`` with points of (transformed) target point set.
            ys: Tensors of shape ``(N, Y, D)`` with points of other point sets.

        Returns:
            Point set distance.

        """
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class ParamsLoss(paddle.nn.Layer, metaclass=ABCMeta):
    r"""Regularization loss based on model parameters."""

    @abstractmethod
    def forward(self, params: paddle.Tensor) -> paddle.Tensor:
        r"""Evaluate loss term for given model parameters."""
        raise NotImplementedError(f"{type(self).__name__}.forward()")

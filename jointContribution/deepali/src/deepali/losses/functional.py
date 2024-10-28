r"""Loss functions, evaluation metrics, and related utilities."""
import itertools
import math
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import Union

import paddle
from deepali.core.enum import FlowDerivativeKeys
from deepali.core.enum import SpatialDerivativeKeys
from deepali.core.flow import denormalize_flow
from deepali.core.flow import divergence
from deepali.core.flow import flow_derivatives
from deepali.core.flow import spatial_derivatives
from deepali.core.grid import Grid
from deepali.core.image import avg_pool
from deepali.core.image import dot_channels
from deepali.core.image import rand_sample
from deepali.core.pointset import transform_grid
from deepali.core.pointset import transform_points
from deepali.core.tensor import as_one_hot_tensor
from deepali.core.tensor import move_dim
from deepali.core.typing import Array
from deepali.core.typing import Scalar
from deepali.core.typing import ScalarOrTuple
from deepali.utils import paddle_aux  # noqa
from paddle import Tensor

__all__ = (
    "balanced_binary_cross_entropy_with_logits",
    # "binary_cross_entropy_with_logits",
    "label_smoothing",
    "dice_score",
    "dice_loss",
    "kld_loss",
    "lcc_loss",
    "mae_loss",
    "mse_loss",
    "ncc_loss",
    "ssd_loss",
    "mi_loss",
    "grad_loss",
    "bending_loss",
    "bending_energy",
    "be_loss",
    "bspline_bending_loss",
    "bspline_bending_energy",
    "bspline_be_loss",
    "curvature_loss",
    "diffusion_loss",
    "divergence_loss",
    "elasticity_loss",
    "focal_loss_with_logits",
    "total_variation_loss",
    "tv_loss",
    "tversky_index",
    "tversky_index_with_logits",
    "tversky_loss",
    "tversky_loss_with_logits",
    "inverse_consistency_loss",
    "masked_loss",
    "reduce_loss",
    "wlcc_loss",
)


class ElementwiseLoss(Protocol):
    r"""Type annotation of a eleemntwise loss function."""

    def __call__(
        self, input: paddle.Tensor, target: paddle.Tensor, reduction: str = "mean"
    ) -> paddle.Tensor:
        ...


def label_smoothing(
    labels: paddle.Tensor,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    alpha: float = 0.1,
) -> paddle.Tensor:
    r"""Apply label smoothing to target labels.

    Implements label smoothing as proposed by Muller et al., (2019) in https://arxiv.org/abs/1906.02629v2.

    Args:
        labels: Scalar target labels or one-hot encoded class probabilities.
        num_classes: Number of target labels. If ``None``, use maximum value in ``target`` plus 1
            when a scalar label map is given.
        ignore_index: Ignore index to be kept during the expansion. The locations of the index
            value in the labels image is stored in the corresponding locations across all channels
            so that this location can be ignored across all channels later, e.g. in Dice computation.
            This argument must be ``None`` if ``labels`` has ``C == num_channels``.
        alpha: Label smoothing factor in [0, 1]. If zero, no label smoothing is done.

    Returns:
        Multi-channel tensor of smoothed target class probabilities.

    """
    if not isinstance(labels, paddle.Tensor):
        raise TypeError("label_smoothing() 'labels' must be Tensor")
    if labels.ndim < 4:
        raise ValueError("label_smoothing() 'labels' must be tensor of shape (N, C, ..., X)")
    if tuple(labels.shape)[1] == 1:
        target = as_one_hot_tensor(labels, num_classes, ignore_index=ignore_index, dtype="float32")
    else:
        target = labels.astype(dtype="float32")
    if alpha > 0:
        target = (1 - alpha) * target + alpha * (1 - target) / (target.shape[1] - 1)
    return target


def balanced_binary_cross_entropy_with_logits(
    logits: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Balanced binary cross entropy.

    Shruti Jadon (2020) A survey of loss functions for semantic segmentation.
    https://arxiv.org/abs/2006.14822

    Args:
        logits: Logits of binary predictions as tensor of shape ``(N, 1, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, 1, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, 1, ..., X)``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Balanced binary cross entropy (bBCE). If ``reduction="none"``, the returned tensor has shape
        ``(N, 1, ..., X)`` with bBCE for each element. Otherwise, it is reduced into a scalar.

    """
    if logits.ndim < 3 or tuple(logits.shape)[1] != 1:
        raise ValueError(
            "balanced_binary_cross_entropy_with_logits() 'logits' must have shape (N, 1, ..., X)"
        )
    if target.ndim < 3 or tuple(target.shape)[1] != 1:
        raise ValueError(
            "balanced_binary_cross_entropy_with_logits() 'target' must have shape (N, 1, ..., X)"
        )
    if tuple(logits.shape)[0] != tuple(target.shape)[0]:
        raise ValueError(
            "balanced_binary_cross_entropy_with_logits() 'logits' and 'target' must have matching batch size"
        )
    neg_weight = (
        target.flatten(start_axis=1).mean(axis=-1).reshape((-1,) + (1,) * (target.ndim - 1))
    )
    pos_weight = 1 - neg_weight
    log_y_pred: Tensor = paddle.nn.functional.log_sigmoid(x=logits)
    loss_pos = -log_y_pred.mul(target)
    loss_neg = logits.sub(log_y_pred).mul(1 - target)
    loss = loss_pos.mul(pos_weight).add(loss_neg.mul(neg_weight))
    loss = masked_loss(loss, weight, "balanced_binary_cross_entropy_with_logits", inplace=True)
    loss = reduce_loss(loss, reduction=reduction)
    return loss


def dice_score(
    input: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Soft Dice similarity of multi-channel classification result.

    Args:
        input: Normalized logits of binary predictions as tensor of shape ``(N, C, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, C, ..., X)``.
        epsilon: Small constant used to avoid division by zero.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Dice similarity coefficient (DSC). If ``reduction="none"``, the returned tensor has shape
        ``(N, C)`` with DSC for each batch. Otherwise, the DSC scores are reduced into a scalar.

    """
    if not isinstance(input, paddle.Tensor):
        raise TypeError("dice_score() 'input' must be paddle.Tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError("dice_score() 'target' must be paddle.Tensor")
    if input.dim() < 3:
        raise ValueError("dice_score() 'input' must be tensor of shape (N, C, ..., X)")
    if tuple(input.shape) != tuple(target.shape):
        raise ValueError("dice_score() 'input' and 'target' must have identical shape")
    y_pred = input.astype(dtype="float32")
    y = target.astype(dtype="float32")
    intersection = dot_channels(y_pred, y, weight=weight)
    denominator = dot_channels(y_pred, y_pred, weight=weight) + dot_channels(y, y, weight=weight)
    loss = (
        intersection.multiply_(y=paddle.to_tensor(2))
        .add_(y=paddle.to_tensor(epsilon))
        .div(denominator.add_(y=paddle.to_tensor(epsilon)))
    )
    loss = reduce_loss(loss, reduction)
    return loss


def dice_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""One minus soft Dice similarity of multi-channel classification result.

    Args:
        input: Normalized logits of binary predictions as tensor of shape ``(N, C, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, C, ..., X)``.
        epsilon: Small constant used to avoid division by zero.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        One minus Dice similarity coefficient (DSC). If ``reduction="none"``, the returned tensor has shape
        ``(N, C)`` with DSC for each batch. Otherwise, the DSC scores are reduced into a scalar.

    """
    dsc = dice_score(input, target, weight=weight, epsilon=epsilon, reduction="none")
    loss = reduce_loss(1 - dsc, reduction)
    return loss


def tversky_index(
    input: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    epsilon: float = 1e-15,
    normalize: bool = False,
    binarize: bool = False,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Tversky index as described in https://arxiv.org/abs/1706.05721.

    Args:
        input: Binary predictions as tensor of shape ``(N, 1, ..., X)``
            or multi-class prediction tensor of shape ``(N, C, ..., X)``.
        target: Target labels as tensor of shape ``(N, ..., X)``, binary classification target
            of shape ``(N, 1, ..., X)``, or one-hot encoded tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1|C, ..., X)``..
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Tversky index (TI). If ``reduction="none"``, the returned tensor has shape ``(N, C)``
        with TI for each batch. Otherwise, the TI values are reduced into a scalar.

    """
    if alpha is None and beta is None:
        alpha = beta = 0.5
    elif alpha is None:
        alpha = 1 - beta
    elif beta is None:
        beta = 1 - alpha
    if not isinstance(input, paddle.Tensor):
        raise TypeError("tversky_index() 'input' must be paddle.Tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError("tversky_index() 'target' must be paddle.Tensor")
    if input.ndim < 3 or tuple(input.shape)[1] < 1:
        raise ValueError(
            "tversky_index() 'input' must be have shape (N, 1, ..., X) or (N, C>1, ..., X)"
        )
    if target.ndim < 2 or tuple(target.shape)[1] < 1:
        raise ValueError(
            "tversky_index() 'target' must be have shape (N, ..., X), (N, 1, ..., X), or (N, C>1, ..., X)"
        )
    if tuple(target.shape)[0] != tuple(input.shape)[0]:
        raise ValueError(
            f"tversky_index() 'input' and 'target' batch size must be identical, got {tuple(input.shape)[0]} != {tuple(target.shape)[0]}"
        )
    input = input.astype(dtype="float32")
    if tuple(input.shape)[1] == 1:
        y_pred = input.sigmoid() if normalize else input
    else:
        y_pred = paddle.nn.functional.softmax(input, axis=1) if normalize else input
    if binarize:
        y_pred = y_pred.round()
    num_classes = max(2, tuple(y_pred.shape)[1])
    if target.ndim == input.ndim:
        y = target.astype(y_pred.dtype)
        if tuple(target.shape)[1] == 1:
            if num_classes > 2:
                raise ValueError(
                    f"tversky_index() 'target' has shape (N, 1, ..., X), but 'input' is multi-class prediction (C={num_classes})"
                )
            if tuple(y_pred.shape)[1] == 2:
                start_49 = y_pred.shape[1] + 1 if 1 < 0 else 1
                y_pred = paddle.slice(y_pred, [1], [start_49], [start_49 + 1])
        else:
            if tuple(target.shape)[1] != num_classes:
                raise ValueError(
                    f"tversky_index() 'target' has shape (N, C, ..., X), but C does not match 'input' with C={num_classes}"
                )
            if tuple(y_pred.shape)[1] == 1:
                start_50 = y.shape[1] + 1 if 1 < 0 else 1
                y = paddle.slice(y, [1], [start_50], [start_50 + 1])
        if binarize:
            y = y.round()
    elif target.ndim + 1 == y_pred.ndim:
        if num_classes == 2 and tuple(y_pred.shape)[1] == 1:
            y = target.unsqueeze(axis=1).greater_equal(y=paddle.to_tensor(0.5)).astype(y_pred.dtype)
            if binarize:
                y = y.round()
        else:
            y = as_one_hot_tensor(target, num_classes, dtype=y_pred.dtype)
    else:
        raise ValueError(
            "tversky_index() 'target' must be tensor of shape (N, ..., X) or (N, C, ... X)"
        )
    if tuple(y.shape) != tuple(y_pred.shape):
        raise ValueError("tversky_index() 'input' and 'target' shapes must be compatible")
    if weight is not None:
        if weight.ndim + 1 == y.ndim:
            weight = weight.unsqueeze(axis=1)
        if weight.ndim != y.ndim:
            raise ValueError("tversky_index() 'weight' shape must be (N, ..., X) or (N, C, ..., X)")
        if tuple(weight.shape)[0] != tuple(target.shape)[0]:
            raise ValueError(
                "tversky_index() 'weight' batch size does not match 'input' and 'target'"
            )
        if tuple(weight.shape)[1] == 1:
            weight = weight.tile(repeat_times=(1,) + (num_classes,) + (1,) * (weight.ndim - 2))
        if tuple(weight.shape) != tuple(y.shape):
            raise ValueError(
                "tversky_index() 'weight' shape must be compatible with 'input' and 'target'"
            )
    intersection = dot_channels(y_pred, y, weight=weight)
    fps = dot_channels(y_pred, 1 - y, weight=weight).mul_(alpha)
    fns = dot_channels(1 - y_pred, y, weight=weight).mul_(beta)
    numerator = intersection.add_(y=paddle.to_tensor(epsilon))
    denominator = numerator.add(fps).add(fns)
    loss = numerator.div(denominator)
    loss = reduce_loss(loss, reduction)
    return loss


def tversky_index_with_logits(
    logits: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    epsilon: float = 1e-15,
    binarize: bool = False,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Tversky index as described in https://arxiv.org/abs/1706.05721.

    Args:
        logits: Binary prediction logits as tensor of shape ``(N, 1, ..., X)``.
        target: Target labels as tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Tversky index (TI). If ``reduction="none"``, the returned tensor has shape ``(N, 1)``
        with TI for each batch. Otherwise, the TI values are reduced into a scalar.

    """
    return tversky_index(
        logits,
        target,
        weight=weight,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        normalize=True,
        binarize=binarize,
        reduction=reduction,
    )


def tversky_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: float = 1e-15,
    normalize: bool = False,
    binarize: bool = False,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Tversky loss as described in https://arxiv.org/abs/1706.05721.

    Args:
        input: Normalized logits of binary predictions as tensor of shape ``(N, C, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        gamma: Exponent used for focal Tverksy loss.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        One minus Tversky index (TI) to the power of gamma. If ``reduction="none"``, the returned
        tensor has shape ``(N, C)`` with the loss for each batch. Otherwise, a scalar is returned.

    """
    ti = tversky_index(
        input,
        target,
        weight=weight,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        epsilon=epsilon,
        normalize=normalize,
        binarize=binarize,
        reduction="none",
    )
    one = paddle.to_tensor(data=1, dtype=ti.dtype, place=ti.place)
    loss = one.sub(ti)
    if gamma:
        if gamma > 1:
            loss = loss.pow_(y=gamma)
        elif gamma < 1:
            raise ValueError("tversky_loss() 'gamma' must be greater than or equal to 1")
    loss = reduce_loss(loss, reduction)
    return loss


def tversky_loss_with_logits(
    logits: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: float = 1e-15,
    binarize: bool = False,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Tversky loss as described in https://arxiv.org/abs/1706.05721.

    Args:
        logits: Binary prediction logits as tensor of shape ``(N, 1, ..., X)``.
        target: Target labels as tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        gamma: Exponent used for focal Tverksy loss.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        One minus Tversky index (TI) to the power of gamma. If ``reduction="none"``, the returned
        tensor has shape ``(N, C)`` with the loss for each batch. Otherwise, a scalar is returned.

    """
    return tversky_loss(
        logits,
        target,
        weight=weight,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        epsilon=epsilon,
        normalize=True,
        binarize=binarize,
        reduction=reduction,
    )


def focal_loss_with_logits(
    input: paddle.Tensor,
    target: paddle.Tensor,
    weight: Optional[paddle.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        input: Logits of the predictions for each example.
        target: A tensor with the same shape as ``input``. Stores the binary classification
            label for each element in inputs (0 for the negative class and 1 for the positive class).
        weight: Multiplicative mask tensor with same shape as ``input``.
        alpha: Weighting factor in [0, 1] to balance positive vs negative examples or -1 for ignore.
        gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Loss tensor with the reduction option applied.

    """
    bce = paddle.nn.functional.binary_cross_entropy_with_logits(
        logit=input, label=target, reduction="none"
    )
    one = paddle.to_tensor(data=1, dtype=bce.dtype, place=bce.place)
    loss = one.sub(paddle.exp(x=-bce)).pow(y=gamma).mul(bce)
    if alpha >= 0:
        if alpha > 1:
            raise ValueError("focal_loss() 'alpha' must be in [0, 1]")
        loss = target.mul(alpha).add(one.sub(target).mul(1 - alpha)).mul(loss)
    loss = masked_loss(loss, weight, "focal_loss_with_logits", inplace=True)
    loss = reduce_loss(loss, reduction)
    return loss


def kld_loss(mean: paddle.Tensor, logvar: paddle.Tensor, reduction: str = "mean") -> paddle.Tensor:
    r"""Kullback-Leibler divergence in case of zero-mean and isotropic unit variance normal prior distribution.

    Kingma and Welling, Auto-Encoding Variational Bayes, ICLR 2014, https://arxiv.org/abs/1312.6114 (Appendix B).

    """
    loss = (
        mean.square()
        .add_(y=paddle.to_tensor(logvar.exp()))
        .subtract_(y=paddle.to_tensor(1))
        .subtract_(y=paddle.to_tensor(logvar))
    )
    loss = reduce_loss(loss, reduction)
    loss = loss.multiply_(y=paddle.to_tensor(0.5))
    return loss


def ncc_loss(
    source: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Normalized cross correlation.

    Args:
        source: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``source``.
        mask: Multiplicative mask tensor with same shape as ``source``.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" of normalized cross correlation
            of image pairs in batch. If "none", a 1-dimensional tensor is returned
            with length equal the batch size.

    Returns:
        Negative squared normalized cross correlation plus one.

    """
    if not isinstance(source, paddle.Tensor):
        raise TypeError("ncc_loss() 'source' must be tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError("ncc_loss() 'target' must be tensor")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("ncc_loss() 'source' must have same shape as 'target'")
    source = source.reshape(tuple(source.shape)[0], -1).astype(dtype="float32")
    target = target.reshape(tuple(source.shape)[0], -1).astype(dtype="float32")
    source_mean = source.mean(axis=1, keepdim=True)
    target_mean = target.mean(axis=1, keepdim=True)
    x = source.sub(source_mean)
    y = target.sub(target_mean)
    a = x.mul(y).sum(axis=1)
    b = x.square().sum(axis=1)
    c = y.square().sum(axis=1)
    loss = (
        paddle.square_(a)
        .divide_(
            y=paddle.to_tensor(b.multiply_(y=paddle.to_tensor(c)).add_(y=paddle.to_tensor(epsilon)))
        )
        .neg_()
        .add_(y=paddle.to_tensor(1))
    )
    loss = masked_loss(loss, mask, "ncc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def lcc_loss(
    source: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    kernel_size: ScalarOrTuple[int] = 7,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Local normalized cross correlation.

    References:
        Avants et al., 2008, Symmetric Diffeomorphic Image Registration with Cross Correlation:
            Evaluating Automated Labeling of Elderly and Neurodegenerative Brain,
            doi:10.1016/j.media.2007.06.004.

    Args:
        source: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``source``.
        mask: Multiplicative mask tensor with same shape as ``source``.
        kernel_size: Local rectangular window size in number of grid points.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" over all grid points. If "none", output
            tensor shape is equal to the shape of the input tensors given an odd kernel size.

    Returns:
        Negative local normalized cross correlation plus one.

    """
    if not isinstance(source, paddle.Tensor):
        raise TypeError("lcc_loss() 'source' must be tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError("lcc_loss() 'target' must be tensor")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("lcc_loss() 'source' must have same shape as 'target'")

    def local_sum(data: paddle.Tensor) -> paddle.Tensor:
        return avg_pool(data, kernel_size=kernel_size, stride=1, padding=None, divisor_override=1)

    def local_mean(data: paddle.Tensor) -> paddle.Tensor:
        return avg_pool(
            data, kernel_size=kernel_size, stride=1, padding=None, count_include_pad=False
        )

    source = source.astype(dtype="float32")
    target = target.astype(dtype="float32")
    source_mean = local_mean(source)
    target_mean = local_mean(target)
    x = source.sub(source_mean)
    y = target.sub(target_mean)
    a = local_sum(x.mul(y))
    b = local_sum(x.square())
    c = local_sum(y.square())
    loss = (
        paddle.square_(a)
        .divide_(
            y=paddle.to_tensor(b.multiply_(y=paddle.to_tensor(c)).add_(y=paddle.to_tensor(epsilon)))
        )
        .neg_()
        .add_(y=paddle.to_tensor(1))
    )
    loss = masked_loss(loss, mask, "lcc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def wlcc_loss(
    source: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    source_mask: Optional[paddle.Tensor] = None,
    target_mask: Optional[paddle.Tensor] = None,
    kernel_size: ScalarOrTuple[int] = 7,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Weighted local normalized cross correlation.

    References:
        Lewis et al., 2020, Fast Learning-based Registration of Sparse 3D Clinical Images, arXiv:1812.06932.

    Args:
        source: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``source``.
        mask: Multiplicative mask tensor ``w_c`` with same shape as ``target`` and ``source``.
            This tensor is used for computing the weighted local correlation. If ``None`` and
            both ``source_mask`` and ``target_mask`` are given, it is set to the product of these.
            Otherwise, no mask is used to aggregate the local cross correlation values. When both
            ``source_mask`` and ``target_mask`` are ``None``, but ``mask`` is not, then the specified
            ``mask`` is used both as ``source_mask`` and ``target_mask``.
        source_mask: Multiplicative mask tensor ``w_m`` with same shape as ``source``.
            This tensor is used for computing the weighted local ``source`` mean. If ``None``,
            the local mean is computed over all ``source`` elements within each local neighborhood.
        target_mask: Multiplicative mask tensor ``w_f`` with same shape as ``source``.
            This tensor is used for computing the weighted local ``target`` mean. If ``None``,
            the local mean is computed over all ``target`` elements within each local neighborhood.
        kernel_size: Local rectangular window size in number of grid points.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" over all grid points. If "none", output
            tensor shape is equal to the shape of the input tensors given an odd kernel size.

    Returns:
        Negative local normalized cross correlation plus one.

    """
    if not isinstance(source, paddle.Tensor):
        raise TypeError("wlcc_loss() 'source' must be tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError("wlcc_loss() 'target' must be tensor")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError("wlcc_loss() 'source' must have same shape as 'target'")
    for t, t_name, w, w_name in zip(
        [target, source, target],
        ["target", "source", "target"],
        [mask, source_mask, target_mask],
        ["mask", "source_mask", "target_mask"],
    ):
        if w is None:
            continue
        if not isinstance(w, paddle.Tensor):
            raise TypeError(f"wlcc_loss() '{w_name}' must be tensor")
        if tuple(w.shape)[0] not in (1, tuple(t.shape)[0]):
            raise ValueError(
                f"wlcc_loss() '{w_name}' batch size ({tuple(w.shape)[0]}) must be 1 or match '{t_name}' ({tuple(t.shape)[0]})"
            )
        if tuple(w.shape)[1] not in (1, tuple(t.shape)[1]):
            raise ValueError(
                f"wlcc_loss() '{w_name}' number of channels ({tuple(w.shape)[1]}) must be 1 or match '{t_name}' ({tuple(t.shape)[1]})"
            )
        if tuple(w.shape)[2:] != tuple(t.shape)[2:]:
            raise ValueError(
                f"wlcc_loss() '{w_name}' grid shape ({tuple(w.shape)[2:]}) must match '{t_name}' ({tuple(t.shape)[2:]})"
            )

    def local_sum(data: paddle.Tensor) -> paddle.Tensor:
        return avg_pool(data, kernel_size=kernel_size, stride=1, padding=None, divisor_override=1)

    def local_mean(data: paddle.Tensor, weight: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        if weight is None:
            return avg_pool(
                data, kernel_size=kernel_size, stride=1, padding=None, count_include_pad=False
            )
        a = local_sum(data.mul(weight))
        b = local_sum(weight).add_(y=paddle.to_tensor(epsilon))
        return a.divide_(y=paddle.to_tensor(b))

    if mask is not None and source_mask is None and target_mask is None:
        source_mask = mask.astype(dtype="float32")
        target_mask = source_mask
    else:
        if source_mask is not None:
            source_mask = source_mask.astype(dtype="float32")
        if target_mask is not None:
            target_mask = target_mask.astype(dtype="float32")
    source = source.astype(dtype="float32")
    target = target.astype(dtype="float32")
    source_mean = local_mean(source, source_mask)
    target_mean = local_mean(target, target_mask)
    x = source.sub(source_mean)
    y = target.sub(target_mean)
    if mask is None and source_mask is not None and target_mask is not None:
        mask = source_mask.mul(target_mask)
    if mask is not None:
        x = x.multiply_(y=paddle.to_tensor(mask))
        y = y.multiply_(y=paddle.to_tensor(mask))
    a = local_sum(x.mul(y))
    b = local_sum(x.square())
    c = local_sum(y.square())
    loss = (
        paddle.square_(a)
        .divide_(
            y=paddle.to_tensor(b.multiply_(y=paddle.to_tensor(c)).add_(y=paddle.to_tensor(epsilon)))
        )
        .neg_()
        .add_(y=paddle.to_tensor(1))
    )
    loss = masked_loss(loss, mask, name="wlcc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def huber_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "mean",
    delta: float = 1.0,
) -> paddle.Tensor:
    r"""Normalized masked Huber loss.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.
        delta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.

    Returns:
        Masked, aggregated, and normalized Huber loss.

    """

    def loss_fn(
        input: paddle.Tensor, target: paddle.Tensor, reduction: str = "mean"
    ) -> paddle.Tensor:
        return paddle.nn.functional.smooth_l1_loss(
            input=input, label=target, reduction=reduction, delta=delta
        )

    return elementwise_loss(
        "huber_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def smooth_l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> paddle.Tensor:
    r"""Normalized masked smooth L1 loss.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.
        delta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.

    Returns:
        Masked, aggregated, and normalized smooth L1 loss.

    """

    def loss_fn(
        input: paddle.Tensor, target: paddle.Tensor, reduction: str = "mean"
    ) -> paddle.Tensor:
        return (
            paddle.nn.functional.smooth_l1_loss(
                input=input, reduction=reduction, label=target, delta=beta
            )
            / beta
        )

    return elementwise_loss(
        "smooth_l1_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Normalized mean absolute error.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Normalized mean absolute error.

    """

    def loss_fn(
        input: paddle.Tensor, target: paddle.Tensor, reduction: str = "mean"
    ) -> paddle.Tensor:
        return paddle.nn.functional.l1_loss(input=input, label=target, reduction=reduction)

    return elementwise_loss(
        "l1_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def mae_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Normalized mean absolute error.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Normalized mean absolute error.

    """

    def loss_fn(
        input: paddle.Tensor, target: paddle.Tensor, reduction: str = "mean"
    ) -> paddle.Tensor:
        return paddle.nn.functional.l1_loss(input=input, label=target, reduction=reduction)

    return elementwise_loss(
        "mae_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def mse_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Average normalized squared differences.

    This loss is equivalent to `ssd_loss`, except that the default `reduction` is "mean".

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Average normalized squared differences.

    """
    return ssd_loss(input, target, mask=mask, norm=norm, reduction=reduction)


def ssd_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "sum",
) -> paddle.Tensor:
    r"""Sum of normalized squared differences.

    The SSD loss is equivalent to MSE, except that an optional overlap mask is supported and
    that the loss value is optionally multiplied by a normalization constant. Moreover, by default
    the sum instead of the mean of per-element loss values is returned (cf. ``reduction``).
    The value returned by ``max_difference(source, target).square()`` can be used as normalization
    factor, which is equvalent to first normalizing the images to [0, 1].

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Sum of normalized squared differences.

    """
    if not isinstance(input, paddle.Tensor):
        raise TypeError("ssd_loss() 'input' must be tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError("ssd_loss() 'target' must be tensor")
    if tuple(input.shape) != tuple(target.shape):
        raise ValueError("ssd_loss() 'input' must have same shape as 'target'")
    loss = input.sub(target).square()
    loss = masked_loss(loss, mask, "ssd_loss")
    loss = reduce_loss(loss, reduction, mask)
    if norm is not None:
        norm = paddle.to_tensor(data=norm, dtype=loss.dtype, place=loss.place).squeeze()
        if not norm.ndim == 0:
            raise ValueError("ssd_loss() 'norm' must be scalar")
        if norm > 0:
            loss = loss.divide_(y=paddle.to_tensor(norm))
    return loss


def mi_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    num_bins: Optional[int] = None,
    num_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    normalized: bool = False,
) -> paddle.Tensor:
    r"""Calculate mutual information loss using Parzen window density and entropy estimations.

    References:
        Qiu, H., Qin, C., Schuh, A., Hammernik, K.: Learning Diffeomorphic and Modality-invariant
            Registration using B-splines. Medical Imaging with Deep Learning. (2021).
        Thévenaz, P., Unser, M.: Optimization of mutual information for multiresolution image registration.
            IEEE Trans. Image Process. 9, 2083–2099 (2000).

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Region of interest mask with same shape as ``input``.
        vmin: Minimal intensity value the joint and marginal density is estimated.
        vmax: Maximal intensity value the joint and marginal density is estimated.
        num_bins: Number of bin edges to discretize the density estimation.
        num_samples: Number of voxels in the image domain randomly sampled to compute the loss,
            ignored if `sample_ratio` is also set.
        sample_ratio: Ratio of voxels in the image domain randomly sampled to compute the loss.
        normalized: Calculate Normalized Mutual Information instead of Mutual Information if True.

    Returns:
        Negative mutual information. If ``normalized=True``, 2 is added such that the loss value is in [0, 2].

    """
    if target.ndim < 3:
        raise ValueError("mi_loss() 'target' must be tensor of shape (N, C, ..., X)")
    if tuple(input.shape) != tuple(target.shape):
        raise ValueError("ssd_loss() 'input' must have same shape as 'target'")
    if vmin is None:
        vmin = paddle_aux.min(input.min(), target.min()).item()
    if vmax is None:
        vmax = paddle_aux.max(input.max(), target.max()).item()
    if num_bins is None:
        num_bins = 64
    elif num_bins == "auto":
        raise NotImplementedError(
            "mi_loss() automatically setting num_bins based on dynamic range of input"
        )
    shape = tuple(target.shape)
    input = input.flatten(start_axis=2)
    target = target.flatten(start_axis=2)
    if mask is not None:
        if mask.ndim < 3 or tuple(mask.shape)[2:] != shape[2:] or tuple(mask.shape)[1] != 1:
            raise ValueError(
                "mi_loss() 'mask' must be tensor of shape (1|N, 1, ..., X) with spatial dimensions matching 'target'"
            )
        mask = mask.flatten(start_axis=2)
    if sample_ratio is not None:
        if sample_ratio <= 0 or sample_ratio > 1:
            raise ValueError("mi_loss() 'sample_ratio' must be in open-closed interval (0, 1]")
        if num_samples is None:
            num_samples = tuple(target.shape)[2:].size
        num_samples = min(max(1, int(sample_ratio * tuple(target.shape)[2:].size)), num_samples)
    if num_samples is not None:
        input, target = rand_sample([input, target], num_samples, mask=mask, replacement=True)
    elif mask is not None:
        input = input.mul(mask)
        target = target.mul(mask)
    bin_width = (vmax - vmin) / num_bins
    out_1 = paddle.linspace(start=vmin, stop=vmax, num=num_bins)
    out_1.stop_gradient = not False
    bin_center = out_1
    bin_center = bin_center.unsqueeze(axis=1).astype(dtype=input.dtype)
    pw_sdev = bin_width * (1 / (2 * math.sqrt(2 * math.log(2))))
    pw_norm = 1 / math.sqrt(2 * math.pi) * pw_sdev

    def parzen_window_fn(x: paddle.Tensor) -> paddle.Tensor:
        return x.sub(bin_center).square().div(2 * pw_sdev**2).neg().exp().mul(pw_norm)

    pw_input = parzen_window_fn(input)
    pw_target = parzen_window_fn(target)
    hist_joint = pw_input.bmm(
        y=pw_target.transpose(perm=paddle_aux.transpose_aux_func(pw_target.ndim, 1, 2))
    )
    hist_norm = (
        hist_joint.flatten(start_axis=1, stop_axis=-1).sum(axis=1).add_(y=paddle.to_tensor(1e-05))
    )
    p_joint = hist_joint / hist_norm.view(-1, 1, 1)
    p_input = p_joint.sum(axis=2)
    p_target = p_joint.sum(axis=1)
    ent_input = -paddle.sum(x=p_input * paddle.log(x=p_input + 1e-05), axis=1)
    ent_target = -paddle.sum(x=p_target * paddle.log(x=p_target + 1e-05), axis=1)
    ent_joint = -paddle.sum(x=p_joint * paddle.log(x=p_joint + 1e-05), axis=(1, 2))
    if normalized:
        loss = 2 - paddle.mean(x=(ent_input + ent_target) / ent_joint)
    else:
        loss = paddle.mean(x=ent_input + ent_target - ent_joint).neg()
    return loss


def nmi_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    num_bins: Optional[int] = None,
    num_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
) -> paddle.Tensor:
    return mi_loss(
        input,
        target,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        num_bins=num_bins,
        num_samples=num_samples,
        sample_ratio=sample_ratio,
        normalized=True,
    )


def grad_loss(
    u: paddle.Tensor,
    p: Union[int, float] = 2,
    q: Optional[Union[int, float]] = 1,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Loss term based on p-norm of spatial gradient of vector fields.

    The ``p`` and ``q`` parameters can be used to specify which norm to compute, i.e., ``sum(abs(du)**p)**q``,
    where ``du`` are the 1st order spatial derivative of the input vector fields ``u`` computed using a finite
    difference scheme and optionally normalized using a specified grid ``spacing``.

    This regularization loss is the basis, for example, for total variation and diffusion penalties.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        p: The order of the gradient norm. When ``p = 0``, the partial derivatives are summed up.
        q: Power parameter of gradient norm. If ``None``, then ``q = 1 / p``. If ``q == 0``, the
            absolute value of the sum of partial derivatives is computed at each grid point.
        mode: Method used to approximate :func:`spatial_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Spatial gradient loss of vector fields.

    """
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "grad_loss() not implemented for linear transformation and 'reduction'='none'"
            )
        return paddle.to_tensor(data=0, dtype=u.dtype, place=u.place)
    D = tuple(u.shape)[1]
    if u.ndim - 2 != D:
        raise ValueError("grad_loss() 'u' must be tensor of shape (N, D, ..., X)")
    if q is None:
        q = 1.0 / p
    if spacing is None:
        spacing = tuple(reversed([(2 / (n - 1)) for n in tuple(u.shape)[2:]]))
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    which = SpatialDerivativeKeys.all(spatial_dims=D, order=1)
    deriv = spatial_derivatives(u, which=which, **kwargs)
    if p == 1:
        deriv = {k: v.abs_() for k, v in deriv.items()}
    elif p != 0:
        if p % 2 == 0:
            deriv = {k: v.pow_(y=p) for k, v in deriv.items()}
        else:
            deriv = {k: v.abs_().pow_(y=p) for k, v in deriv.items()}
    loss: Optional[paddle.Tensor] = None
    for value in deriv.values():
        value = value.sum(axis=1, keepdim=True)
        loss = value if loss is None else loss.add_(y=paddle.to_tensor(value))
    assert loss is not None
    if q == 0:
        loss.abs_()
    elif q != 1:
        loss.pow_(y=q)
    loss = reduce_loss(loss, reduction)
    return loss


def bending_loss(
    u: paddle.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Bending energy of vector fields.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        mode: Method used to approximate :func:`flow_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Bending energy.

    """
    if u.ndim < 4:
        if reduction == "none":
            raise NotImplementedError(
                "bending_energy() not implemented for linear transformation and 'reduction'='none'"
            )
        return paddle.to_tensor(data=0, dtype=u.dtype, place=u.place)
    D = tuple(u.shape)[1]
    if u.ndim - 2 != D:
        raise ValueError("bending_energy() 'u' must be tensor of shape (N, D, ..., X)")
    kwargs = dict(mode=mode or "sobel", sigma=sigma, spacing=spacing, stride=stride)
    which = FlowDerivativeKeys.all(spatial_dims=D, order=2)
    which = sorted(FlowDerivativeKeys.unique(which))
    deriv = flow_derivatives(u, which=which, **kwargs)
    loss: Optional[paddle.Tensor] = None
    for key, value in deriv.items():
        value = paddle.square(value)
        if FlowDerivativeKeys.is_mixed(key):
            value = value.multiply_(y=paddle.to_tensor(2.0))
        loss = value if loss is None else loss.add_(y=paddle.to_tensor(value))
    assert loss is not None
    loss = reduce_loss(loss, reduction)
    return loss


be_loss = bending_loss
bending_energy = bending_loss


def bspline_bending_loss(
    data: paddle.Tensor, stride: ScalarOrTuple[int] = 1, reduction: str = "mean"
) -> paddle.Tensor:
    r"""Evaluate bending energy of cubic B-spline function, e.g., spatial free-form deformation.

    Deprecated, use :func:`bending_loss()` with ``mode='bspline'`` instead.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, C, ..., X)``.
        stride: Number of points between control points at which to evaluate bending energy, plus one.
            If a sequence of values is given, these must be the strides for the different spatial
            dimensions in the order ``(sx, ...)``. A stride of 1 is equivalent to evaluating bending
            energy only at the usually coarser resolution of the control point grid. It should be noted
            that the stride need not match the stride used to densely sample the spline deformation field
            at a given fixed target image resolution.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Bending energy of cubic B-spline.

    """
    return bending_loss(data, mode="bspline", stride=stride, reduction=reduction)


bspline_be_loss = bspline_bending_loss
bspline_bending_energy = bspline_bending_loss


def curvature_loss(
    u: paddle.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Loss term based on unmixed 2nd order spatial derivatives of vector fields.

    References:
        Fischer & Modersitzki (2003). Curvature based image registration.
            Journal Mathematical Imaging and Vision, 18(1), 81–85.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        mode: Method used to approximate :func:`flow_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Curvature loss of vector fields.

    """
    if u.ndim < 4:
        if reduction == "none":
            raise NotImplementedError(
                "curvature_loss() not implemented for linear transformation and reduction='none'"
            )
        return paddle.to_tensor(data=0, dtype=u.dtype, place=u.place, stop_gradient=u.stop_gradient)
    N = tuple(u.shape)[0]
    D = tuple(u.shape)[1]
    if u.ndim - 2 != D:
        raise ValueError(f"curvature_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    kwargs = dict(mode=mode or "sobel", sigma=sigma, spacing=spacing, stride=stride)
    which = FlowDerivativeKeys.curvature(spatial_dims=D)
    deriv = flow_derivatives(u, which=which, **kwargs)
    shape = tuple(deriv["du/dxx"].shape)
    loss = paddle.zeros(shape=(N, D) + shape[2:], dtype=u.dtype)
    for i, j in itertools.product(range(D), repeat=2):
        start_51 = loss.shape[1] + i if i < 0 else i
        paddle.slice(loss, [1], [start_51], [start_51 + 1]).add_(
            y=paddle.to_tensor(deriv[FlowDerivativeKeys.symbol(i, j, j)])
        )
    loss = paddle.square(loss).sum(axis=1, keepdim=True)
    loss = reduce_loss(loss, reduction)
    loss = loss.multiply_(y=paddle.to_tensor(0.5))
    loss.stop_gradient = u.stop_gradient
    return loss


def diffusion_loss(
    u: paddle.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Diffusion regularization loss.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        mode: Method used to approximate :func:`flow_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Diffusion regularization loss of vector fields.

    """
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    loss = grad_loss(u, p=2, q=1, reduction=reduction, **kwargs)
    return loss.multiply_(y=paddle.to_tensor(0.5))


def divergence_loss(
    u: paddle.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Loss term encouraging divergence-free vector fields.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        mode: Method used to approximate :func:`flow_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Divergence loss of vector fields.

    """
    if u.ndim < 4:
        if reduction == "none":
            raise NotImplementedError(
                "divergence_loss() not implemented for linear transformation and reduction='none'"
            )
        return paddle.to_tensor(data=0, dtype=u.dtype, place=u.place)
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    loss = divergence(u, **kwargs).square_()
    loss = reduce_loss(loss, reduction)
    return loss.multiply_(y=paddle.to_tensor(0.5))


def lame_parameters(
    material_name: Optional[str] = None,
    first_parameter: Optional[float] = None,
    second_parameter: Optional[float] = None,
    shear_modulus: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    youngs_modulus: Optional[float] = None,
) -> Tuple[float, float]:
    r"""Get Lame parameters of linear elasticity given different quantities.

    Args:
        material_name: Name of material preset. Cannot be used in conjunction with other arguments.
        first_parameter: Lame's first parameter.
        second_parameter: Lame's second parameter, i.e., shear modulus.
        shear_modulus: Shear modulus, i.e., Lame's second parameter.
        poissons_ratio: Poisson's ratio.
        youngs_modulus: Young's modulus.

    Returns:
        lambda: Lame's first parameter.
        mu: Lame's second parameter.

    """
    RUBBER_POISSONS_RATIO = 0.4999
    RUBBER_SHEAR_MODULUS = 0.0006
    # Derive unspecified Lame parameters from any combination of two given quantities
    # (cf. conversion table at https://en.wikipedia.org/wiki/Young%27s_modulus#External_links)
    kwargs = {
        name: value
        for name, value in zip(
            [
                "first_parameter",
                "second_parameter",
                "shear_modulus",
                "poissons_ratio",
                "youngs_modulus",
            ],
            [first_parameter, second_parameter, poissons_ratio, youngs_modulus, shear_modulus],
        )
        if value is not None
    }
    # Default values for different materials (cf. Wikipedia pages for Poisson's ratio and shear modulus)
    if material_name:
        if kwargs:
            raise ValueError(
                "lame_parameters() 'material_name' cannot be specified in combination with other quantities"
            )
        if material_name == "rubber":
            poissons_ratio = RUBBER_POISSONS_RATIO
            shear_modulus = RUBBER_SHEAR_MODULUS
        else:
            raise ValueError(f"lame_parameters() unknown 'material_name': {material_name}")
    elif len(kwargs) != 2:
        raise ValueError(
            "lame_parameters() specify 'material_name' or exactly two parameters, got: "
            + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        )
    if second_parameter is None:
        second_parameter = shear_modulus
    elif shear_modulus is None:
        shear_modulus = second_parameter
    else:
        raise ValueError(
            "lame_parameters() 'second_parameter' and 'shear_modulus' are mutually exclusive"
        )
    if first_parameter is None:
        if shear_modulus is None:
            if poissons_ratio is not None and youngs_modulus is not None:
                first_parameter = (
                    poissons_ratio * youngs_modulus / ((1 + poissons_ratio)(1 - 2 * poissons_ratio))
                )
                second_parameter = youngs_modulus / (2 * (1 + poissons_ratio))
        elif youngs_modulus is None:
            if poissons_ratio is None:
                poissons_ratio = RUBBER_POISSONS_RATIO
            first_parameter = 2 * shear_modulus * poissons_ratio / (1 - 2 * poissons_ratio)
        else:
            first_parameter = (
                shear_modulus
                * (youngs_modulus - 2 * shear_modulus)
                / (3 * shear_modulus - youngs_modulus)
            )
    elif second_parameter is None:
        if youngs_modulus is None:
            if poissons_ratio is None:
                poissons_ratio = RUBBER_POISSONS_RATIO
            second_parameter = first_parameter * (1 - 2 * poissons_ratio) / (2 * poissons_ratio)
        else:
            r = math.sqrt(
                youngs_modulus**2
                + 9 * first_parameter**2
                + 2 * youngs_modulus * first_parameter
            )
            second_parameter = youngs_modulus - 3 * first_parameter + r / 4
    if first_parameter is None or second_parameter is None:
        raise NotImplementedError(
            "lame_parameters() deriving Lame parameters from: "
            + ", ".join(f"'{name}'" for name in kwargs.keys())
        )
    if first_parameter < 0:
        raise ValueError("lame_parameter() 'first_parameter' is negative")
    elif first_parameter < 1e-9:
        first_parameter = 0
    if second_parameter < 0:
        raise ValueError("lame_parameter() 'second_parameter' is negative")
    elif second_parameter < 1e-9:
        second_parameter = 0
    return first_parameter, second_parameter


def elasticity_loss(
    u: paddle.Tensor,
    material_name: Optional[str] = None,
    first_parameter: Optional[float] = None,
    second_parameter: Optional[float] = None,
    shear_modulus: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    youngs_modulus: Optional[float] = None,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Loss term based on Navier-Cauchy PDE of linear elasticity.

    References:
        Fischer & Modersitzki, 2004, A unified approach to fast image registration and a new
            curvature based registration technique.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        material_name: Name of material preset. Cannot be used in conjunction with other arguments.
        first_parameter: Lame's first parameter.
        second_parameter: Lame's second parameter, i.e., shear modulus.
        shear_modulus: Shear modulus, i.e., Lame's second parameter.
        poissons_ratio: Poisson's ratio.
        youngs_modulus: Young's modulus.
        mode: Method used to approximate :func:`flow_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Linear elasticity loss of vector field.

    """
    lambd, mu = lame_parameters(
        material_name=material_name,
        first_parameter=first_parameter,
        second_parameter=second_parameter,
        shear_modulus=shear_modulus,
        poissons_ratio=poissons_ratio,
        youngs_modulus=youngs_modulus,
    )
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "elasticity_loss() not implemented for linear transformation and reduction='none'"
            )
        return paddle.to_tensor(data=0, dtype=u.dtype, place=u.place)
    N = tuple(u.shape)[0]
    D = tuple(u.shape)[1]
    if u.ndim - 2 != D:
        raise ValueError(f"elasticity_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    which = FlowDerivativeKeys.jacobian(spatial_dims=D)
    deriv = flow_derivatives(u, which=which, **kwargs)
    loss = paddle.zeros(shape=(N, 1) + tuple(u.shape)[2:], dtype=u.dtype)
    if lambd != 0:
        for i in range(D):
            loss = loss.add_(y=paddle.to_tensor(deriv[FlowDerivativeKeys.symbol(i, i)]))
        loss = paddle.square_(loss).multiply_(y=paddle.to_tensor(lambd / 2))
    if mu != 0:
        for j, k in itertools.product(range(D), repeat=2):
            djk = deriv[FlowDerivativeKeys.symbol(j, k)]
            dkj = deriv[FlowDerivativeKeys.symbol(k, j)]
            loss = loss.add_(
                y=paddle.to_tensor(
                    paddle.square_(djk.add(dkj)).multiply_(y=paddle.to_tensor(mu / 4))
                )
            )
    loss = reduce_loss(loss, reduction)
    return loss


def total_variation_loss(
    u: paddle.Tensor,
    mode: Optional[str] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Union[Scalar, Array]] = None,
    stride: Optional[ScalarOrTuple] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Total variation regularization loss.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        mode: Method used to approximate :func:`flow_derivatives()`.
        sigma: Standard deviation of Gaussian in grid units used to smooth vector field.
        spacing: Step size to use when computing finite differences.
        stride: Number of output grid points between control points plus one for ``mode='bspline'``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Total variation regularization loss of vector fields.

    """
    kwargs = dict(mode=mode, sigma=sigma, spacing=spacing, stride=stride)
    return grad_loss(u, p=1, q=1, reduction=reduction, **kwargs)


tv_loss = total_variation_loss


def inverse_consistency_loss(
    forward: paddle.Tensor,
    inverse: paddle.Tensor,
    grid: Optional[Grid] = None,
    margin: Union[int, float] = 0,
    mask: Optional[paddle.Tensor] = None,
    units: str = "cube",
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Evaluate inverse consistency error.

    This function expects forward and inverse coordinate maps to be with respect to the unit cube
    of side length 2 as defined by the domain and codomain ``grid`` (see also ``Grid.axes()``).

    Args:
        forward: Tensor representation of spatial transformation.
        inverse: Tensor representation of inverse transformation.
        grid: Coordinate domain and codomain of forward transformation.
        margin: Number of ``grid`` points to ignore when computing mean error. If type of the
            argument is ``int``, this number of points are dropped at each boundary in each dimension.
            If a ``float`` value is given, it must be in [0, 1) and denote the percentage of sampling
            points to drop at each border. Inverse consistency of points near the domain boundary is
            affected by extrapolation and excluding these may be preferrable. See also ``mask``.
        mask: Foreground mask as tensor of shape ``(N, 1, ..., X)`` with size matching ``forward``.
            Inverse consistency errors at target grid points with a zero mask value are ignored.
        units: Compute mean inverse consistency error in specified units: ``cube`` with respect to
            normalized grid cube coordinates, ``voxel`` in voxel units, or in ``world`` units (mm).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Inverse consistency error.

    """
    if not isinstance(forward, paddle.Tensor):
        raise TypeError("inverse_consistency_loss() 'forward' must be tensor")
    if not isinstance(inverse, paddle.Tensor):
        raise TypeError("inverse_consistency_loss() 'inverse' must be tensor")
    if not isinstance(margin, (int, float)):
        raise TypeError("inverse_consistency_loss() 'margin' must be int or float")
    if grid is None:
        if forward.ndim < 4:
            if inverse.ndim < 4:
                raise ValueError(
                    "inverse_consistency_loss() 'grid' required when both transforms are affine"
                )
            grid = Grid(shape=tuple(inverse.shape)[2:])
        else:
            grid = Grid(shape=tuple(forward.shape)[2:])
    x = grid.coords(dtype=forward.dtype, device=forward.place).unsqueeze(axis=0)
    y = transform_grid(forward, x, align_corners=grid.align_corners())
    y = transform_points(inverse, y, align_corners=grid.align_corners())
    error = y - x
    if mask is not None:
        if not isinstance(mask, paddle.Tensor):
            raise TypeError("inverse_consistency_loss() 'mask' must be tensor")
        if mask.ndim != grid.ndim + 2:
            raise ValueError(
                f"inverse_consistency_loss() 'mask' must be {grid.ndim + 2}-dimensional"
            )
        if tuple(mask.shape)[1] != 1:
            raise ValueError("inverse_consistency_loss() 'mask' must have shape (N, 1, ..., X)")
        if tuple(mask.shape)[0] != 1 and tuple(mask.shape)[0] != tuple(error.shape)[0]:
            raise ValueError(
                f"inverse_consistency_loss() 'mask' batch size must be 1 or {tuple(error.shape)[0]}"
            )
        error[move_dim(mask == 0, 1, -1).expand_as(error)] = 0
    # Discard error at grid boundary
    if margin > 0:
        if isinstance(margin, float):
            if margin < 0 or margin >= 1:
                raise ValueError(
                    f"inverse_consistency_loss() 'margin' must be in [0, 1), got {margin}"
                )
            m = [int(margin * n) for n in tuple(grid.shape)]
        else:
            m = [max(0, int(margin))] * grid.ndim
        subgrid = tuple(reversed([slice(i, n - i) for i, n in zip(m, tuple(grid.shape))]))
        error = error[(slice(0, tuple(error.shape)[0]),) + subgrid + (slice(0, grid.ndim),)]
    if units in ("voxel", "world"):
        error = denormalize_flow(error, size=tuple(grid.shape), channels_last=True)
        if units == "world":
            error *= grid.spacing().to(error)
    error: Tensor = error.norm(p=2, axis=-1)
    if reduction != "none":
        count = error.size
        error = error.sum()
        if reduction == "mean" and mask is not None:
            count = (mask != 0).sum()
        error /= count
    return error


def elementwise_loss(
    name: str,
    loss_fn: ElementwiseLoss,
    input: paddle.Tensor,
    target: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    norm: Optional[Union[float, paddle.Tensor]] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    r"""Evaluate, aggregate, and normalize elementwise loss, optionally within masked region.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Aggregated normalized loss value.

    """
    if not isinstance(input, paddle.Tensor):
        raise TypeError(f"{name}() 'input' must be tensor")
    if not isinstance(target, paddle.Tensor):
        raise TypeError(f"{name}() 'target' must be tensor")
    if tuple(input.shape) != tuple(target.shape):
        raise ValueError(f"{name}() 'input' must have same shape as 'target'")
    if mask is None:
        loss = loss_fn(input, target, reduction=reduction)
    else:
        loss = loss_fn(input, target, reduction="none")
        loss = masked_loss(loss, mask, name)
        loss = reduce_loss(loss, reduction, mask)
    if norm is not None:
        norm = paddle.to_tensor(data=norm, dtype=loss.dtype, place=loss.place).squeeze()
        if not norm.ndim == 0:
            raise ValueError(f"{name}() 'norm' must be scalar")
        if norm > 0:
            loss = loss.divide_(y=paddle.to_tensor(norm))
    return loss


def masked_loss(
    loss: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
    name: Optional[str] = None,
    inplace: bool = False,
) -> paddle.Tensor:
    r"""Multiply loss with an optionally specified spatial mask."""
    if mask is None:
        return loss
    if not name:
        name = "masked_loss"
    if not isinstance(mask, paddle.Tensor):
        raise TypeError(f"{name}() 'mask' must be tensor")
    if tuple(mask.shape)[0] != 1 and tuple(mask.shape)[0] != tuple(loss.shape)[0]:
        raise ValueError(f"{name}() 'mask' must have same batch size as 'target' or batch size 1")
    if tuple(mask.shape)[1] != 1 and tuple(mask.shape)[1] != tuple(loss.shape)[1]:
        raise ValueError(f"{name}() 'mask' must have same number of channels as 'target' or only 1")
    if tuple(mask.shape)[2:] != tuple(loss.shape)[2:]:
        raise ValueError(f"{name}() 'mask' must have same spatial shape as 'target'")
    if inplace:
        loss = loss.multiply_(y=paddle.to_tensor(mask))
    else:
        loss = loss.mul(mask)
    return loss


def reduce_loss(
    loss: paddle.Tensor, reduction: str = "mean", mask: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    r"""Reduce loss computed at each grid point."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduce_loss() 'reduction' must be 'mean', 'sum' or 'none'")
    if reduction == "none":
        return loss
    if mask is None:
        return loss.mean() if reduction == "mean" else loss.sum()
    value = loss.sum()
    if reduction == "mean":
        numel = mask.expand_as(y=loss).sum()
        value = value.divide_(y=paddle.to_tensor(numel))
    return value

import math
from typing import Optional

import paddle
from paddle_utils import add_tensor_methods

__all__ = [
    "focal_loss_with_logits",
    "softmax_focal_loss_with_logits",
    "soft_jaccard_score",
    "soft_dice_score",
    "wing_loss",
]

add_tensor_methods()


def focal_loss_with_logits(
    output: paddle.Tensor,
    target: paddle.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-06,
    ignore_index=None,
) -> paddle.Tensor:
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the models)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.astype(dtype=output.dtype)
    p = paddle.nn.functional.sigmoid(x=output)
    ce_loss = paddle.nn.functional.binary_cross_entropy_with_logits(
        logit=output, label=target, reduction="none"
    )
    pt = p * target + (1 - p) * (1 - target)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(y=gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(y=gamma)
        focal_term = paddle.masked_fill(
            x=focal_term, mask=pt < reduced_threshold, value=1
        )
    loss = focal_term * ce_loss
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if ignore_index is not None:
        ignore_mask = target.equal(y=ignore_index)
        loss = paddle.masked_fill(x=loss, mask=ignore_mask, value=0)
        if normalized:
            focal_term = paddle.masked_fill(x=focal_term, mask=ignore_mask, value=0)
    if normalized:
        norm_factor = focal_term.sum(dtype="float32").clamp_min(eps)
        loss /= norm_factor
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum(dtype="float32")
    if reduction == "batchwise_mean":
        loss = loss.sum(axis=0, dtype="float32")
    return loss


def softmax_focal_loss_with_logits(
    output: paddle.Tensor,
    target: paddle.Tensor,
    gamma: float = 2.0,
    reduction="mean",
    normalized=False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-06,
) -> paddle.Tensor:
    """
    Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
    log_softmax = paddle.nn.functional.log_softmax(x=output, axis=1)
    loss = paddle.nn.functional.nll_loss(
        input=log_softmax, label=target, reduction="none"
    )
    pt = paddle.exp(x=-loss)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(y=gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(y=gamma)
        focal_term[pt < reduced_threshold] = 1
    loss = focal_term * loss
    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss = loss / norm_factor
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(axis=0)
    return loss


def soft_jaccard_score(
    output: paddle.Tensor,
    target: paddle.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-07,
    dims=None,
) -> paddle.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :param dims:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means
            any number of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert tuple(output.shape) == tuple(target.shape)
    if dims is not None:
        intersection = paddle.sum(x=output * target, axis=dims)
        cardinality = paddle.sum(x=output + target, axis=dims)
    else:
        intersection = paddle.sum(x=output * target)
        cardinality = paddle.sum(x=output + target)
    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / paddle.clip(union + smooth, min=eps)
    return jaccard_score


def soft_dice_score(
    output: paddle.Tensor,
    target: paddle.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-07,
    dims=None,
) -> paddle.Tensor:
    """

    :param output:
    :param target:
    :param smooth:
    :param eps:
    :return:

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    """
    assert tuple(output.shape) == tuple(target.shape)
    if dims is not None:
        intersection = paddle.sum(x=output * target, axis=dims)
        cardinality = paddle.sum(x=output + target, axis=dims)
    else:
        intersection = paddle.sum(x=output * target)
        cardinality = paddle.sum(x=output + target)
    denominator = paddle.clip(cardinality + smooth, min=eps)
    dice_score = (2.0 * intersection + smooth) / denominator
    return dice_score


def wing_loss(
    output: paddle.Tensor,
    target: paddle.Tensor,
    width=5,
    curvature=0.5,
    reduction="mean",
):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param output:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target.astype("float32") - output).abs().astype("float32")
    small_loss = width * paddle.log(1 + diff_abs / curvature)
    C = width - width * math.log(1 + width / curvature)
    mask_small = diff_abs < width
    loss = diff_abs.clone()
    loss = paddle.where(mask_small, small_loss, loss)
    loss = paddle.where(~mask_small, loss - C, loss)

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


def label_smoothed_nll_loss(
    lprobs: paddle.Tensor,
    target: paddle.Tensor,
    epsilon: float,
    ignore_index=None,
    reduction="mean",
    dim=-1,
) -> paddle.Tensor:
    """

    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(axis=dim)
    if ignore_index is not None:
        pad_mask = target.equal(y=ignore_index)
        target = paddle.where(pad_mask, paddle.zeros_like(target), target)
        nll_loss = -lprobs.take_along_axis(axis=dim, indices=target)
        smooth_loss = -lprobs.sum(axis=dim, keepdim=True)
        nll_loss = paddle.where(pad_mask, paddle.zeros_like(nll_loss), nll_loss)
        smooth_loss = paddle.where(
            pad_mask, paddle.zeros_like(smooth_loss), smooth_loss
        )
    else:
        nll_loss = -lprobs.take_along_axis(axis=dim, indices=target)
        smooth_loss = -lprobs.sum(axis=dim, keepdim=True)
        nll_loss = nll_loss.squeeze(axis=dim)
        smooth_loss = smooth_loss.squeeze(axis=dim)
    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.shape[dim]
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

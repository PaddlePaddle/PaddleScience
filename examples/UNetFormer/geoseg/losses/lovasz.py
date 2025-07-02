from __future__ import division, print_function
import paddle
from paddle_utils import add_tensor_methods

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
add_tensor_methods()
from typing import Optional, Union

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse
__all__ = ["BinaryLovaszLoss", "LovaszLoss"]


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.astype(dtype="float32").cumsum(axis=0)
    union = gts + (1 - gt_sorted).astype(dtype="float32").cumsum(axis=0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore_index=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(
                    log.unsqueeze(axis=0), lab.unsqueeze(axis=0), ignore_index
                )
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore_index))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Variable, logits at each prediction (between -iinfinity and +iinfinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.astype(dtype="float32") - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = paddle.sort(
        x=errors, axis=0, descending=True
    ), paddle.argsort(x=errors, axis=0, descending=True)
    gt_sorted = paddle.gather(labels, perm)
    grad = _lovasz_grad(gt_sorted)
    loss = paddle.dot(x=paddle.nn.functional.relu(x=errors_sorted), y=grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.reshape([-1])
    labels = labels.reshape([-1])
    if ignore_index is None:
        return scores, labels
    valid = labels != ignore_index
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_softmax(
    probas, labels, classes="present", per_image=False, ignore_index=None
):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore_index: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(
                *_flatten_probas(
                    prob.unsqueeze(axis=0), lab.unsqueeze(axis=0), ignore_index
                ),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(
            *_flatten_probas(probas, labels, ignore_index), classes=classes
        )
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.size == 0:
        return probas * 0.0
    C = probas.shape[1]
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).astype(dtype=probas.dtype)
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, (0)]
        else:
            class_pred = probas[:, (c)]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = paddle.sort(
            x=errors, axis=0, descending=True
        ), paddle.argsort(x=errors, axis=0, descending=True)
        fg_sorted = paddle.gather(fg, perm)
        losses.append(paddle.dot(x=errors_sorted, y=_lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        B, H, W = tuple(probas.shape)
        probas = probas.reshape([B, 1, H, W])
    C = probas.shape[1]
    probas = paddle.moveaxis(x=probas, source=1, destination=-1)
    probas = probas.reshape([-1, C])
    labels = labels.reshape([-1]) 
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class BinaryLovaszLoss(paddle.nn.Layer):
    def __init__(
        self, per_image: bool = False, ignore_index: Optional[Union[int, float]] = None
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_hinge(
            logits, target, per_image=self.per_image, ignore_index=self.ignore_index
        )


class LovaszLoss(paddle.nn.Layer):
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        return _lovasz_softmax(
            logits, target, per_image=self.per_image, ignore_index=self.ignore
        )

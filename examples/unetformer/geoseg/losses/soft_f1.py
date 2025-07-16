from typing import Optional

import paddle
from paddle_utils import add_tensor_methods

__all__ = ["soft_micro_f1", "BinarySoftF1Loss", "SoftF1Loss"]

add_tensor_methods()


def soft_micro_f1(
    preds: paddle.Tensor, targets: paddle.Tensor, eps=1e-06
) -> paddle.Tensor:
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        targets (Tensor): targets array of shape (Num Samples, Num Classes)
        preds (Tensor): probability matrix of shape (Num Samples, Num Classes)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch

    References:
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    """
    tp = paddle.sum(x=preds * targets, axis=0)
    fp = paddle.sum(x=preds * (1 - targets), axis=0)
    fn = paddle.sum(x=(1 - preds) * targets, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + eps)
    loss = 1 - soft_f1
    return loss.mean()


class BinarySoftF1Loss(paddle.nn.Layer):
    def __init__(self, ignore_index: Optional[int] = None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: paddle.Tensor, targets: paddle.Tensor) -> paddle.Tensor:
        targets = targets.view(-1)
        preds = preds.view(-1)
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.size == 0:
                return paddle.to_tensor(data=0, dtype=preds.dtype, place=preds.place)
        preds = paddle.nn.functional.sigmoid(preds).clip(min=self.eps, max=1 - self.eps)
        return soft_micro_f1(preds.view(-1, 1), targets.view(-1, 1))


class SoftF1Loss(paddle.nn.Layer):
    def __init__(self, ignore_index: Optional[int] = None, eps=1e-06):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: paddle.Tensor, targets: paddle.Tensor) -> paddle.Tensor:
        preds = paddle.nn.functional.softmax(preds, axis=1).clip(
            min=self.eps, max=1 - self.eps
        )
        targets = paddle.nn.functional.one_hot(
            num_classes=preds.shape[1], x=targets
        ).astype("int64")
        if self.ignore_index is not None:
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]
            if targets.size == 0:
                return paddle.to_tensor(data=0, dtype=preds.dtype, place=preds.place)
        return soft_micro_f1(preds, targets)

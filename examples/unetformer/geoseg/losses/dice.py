from typing import List

import numpy as np
import paddle
from paddle_utils import add_tensor_methods

from .functional import soft_dice_score

__all__ = ["DiceLoss"]
BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

add_tensor_methods()


def to_tensor(x, dtype=None) -> paddle.Tensor:
    if isinstance(x, paddle.Tensor):
        if dtype is not None:
            x = x.astype(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {"O", "M", "U", "S"}:
        x = paddle.to_tensor(data=x)
        if dtype is not None:
            x = x.astype(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = paddle.to_tensor(data=x)
        if dtype is not None:
            x = x.astype(dtype)
        return x
    raise ValueError("Unsupported input type" + str(type(x)))


class DiceLoss(paddle.nn.Layer):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
        self,
        mode: str = "multiclass",
        classes: List[int] = None,
        log_loss=False,
        from_logits=True,
        smooth: float = 0.0,
        ignore_index=None,
        eps=1e-07,
    ):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert (
                mode != BINARY_MODE
            ), "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype="int64")
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: paddle.Tensor, y_true: paddle.Tensor) -> paddle.Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.shape[0] == y_pred.shape[0]
        if self.from_logits:
            if self.mode == MULTICLASS_MODE:
                y_pred = paddle.nn.functional.log_softmax(y_pred, axis=1).exp()
            else:
                y_pred = paddle.nn.functional.log_sigmoid(x=y_pred).exp()
        bs = y_true.shape[0]
        num_classes = y_pred.shape[1]
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * paddle.cast(mask, dtype="float32")
                y_true = y_true * paddle.cast(mask, dtype="float32")
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    mask = paddle.cast(mask, dtype="float32")
                    y_pred = paddle.cast(
                        y_pred * mask.unsqueeze(axis=1), dtype="float32"
                    )
                    mask_float = paddle.cast(mask, dtype=y_true.dtype)
                    masked_y_true = (y_true * mask_float).astype("int64")
                    y_true = paddle.nn.functional.one_hot(
                        num_classes=num_classes, x=masked_y_true
                    ).astype("int64")
                    mask = paddle.cast(mask, dtype="int64")
                    y_true = y_true.transpose(perm=[0, 2, 1]) * mask.unsqueeze(axis=1)
            else:
                y_true = paddle.nn.functional.one_hot(
                    num_classes=num_classes, x=y_true
                ).astype("int64")
                y_true = y_true.transpose(perm=[0, 2, 1])
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * paddle.cast(mask, dtype="float32")
                y_true = y_true * paddle.cast(mask, dtype="float32")
        scores = soft_dice_score(
            y_pred,
            y_true.astype(dtype=y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )
        if self.log_loss:
            loss = -paddle.log(x=scores.clip(min=self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(axis=dims) > 0
        loss *= mask.astype(loss.dtype)
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()

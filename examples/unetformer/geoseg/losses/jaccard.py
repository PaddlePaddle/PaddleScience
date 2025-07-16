from typing import List

import paddle
import paddle.nn.functional as F
from paddle_utils import add_tensor_methods

from .dice import to_tensor
from .functional import soft_jaccard_score

__all__ = ["JaccardLoss", "BINARY_MODE", "MULTICLASS_MODE", "MULTILABEL_MODE"]
BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

add_tensor_methods()


class JaccardLoss(paddle.nn.Layer):
    """
    Implementation of Jaccard loss for image segmentation task.
    It supports binary, multi-class and multi-label cases.
    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss=False,
        from_logits=True,
        smooth=0,
        eps=1e-07,
    ):
        """

        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()
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
                y_pred = F.log_softmax(y_pred, axis=1).exp()
            else:
                y_pred = F.log_sigmoid(y_pred).exp()
        bs = y_true.shape[0]
        num_classes = y_pred.shape[1]
        dims = 0, 2
        if self.mode == BINARY_MODE:
            y_true = y_true.reshape([bs, 1, -1])
            y_pred = y_pred.reshape([bs, 1, -1])
        if self.mode == MULTICLASS_MODE:
            y_true = y_true.reshape([bs, -1])
            y_pred = y_pred.reshape([bs, num_classes, -1])
            y_true = F.one_hot(y_true, num_classes).astype("int64")
            y_true = y_true.transpose([0, 2, 1])
        if self.mode == MULTILABEL_MODE:
            y_true = y_true.reshape([bs, num_classes, -1])
            y_pred = y_pred.reshape([bs, num_classes, -1])
        scores = soft_jaccard_score(
            y_pred,
            y_true.astype(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )
        if self.log_loss:
            loss = -paddle.log(x=scores.clip(min=self.eps))
        else:
            loss = 1.0 - scores
        mask = y_true.sum(axis=dims) > 0
        loss *= mask.astype(dtype="float32")
        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()

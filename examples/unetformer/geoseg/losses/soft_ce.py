from typing import Optional

import paddle
from paddle_utils import add_tensor_methods

from .functional import label_smoothed_nll_loss

__all__ = ["SoftCrossEntropyLoss"]

add_tensor_methods()


class SoftCrossEntropyLoss(paddle.nn.Layer):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: float = 0.0,
        ignore_index: Optional[int] = -100,
        dim=1,
    ):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        log_prob = paddle.nn.functional.log_softmax(x=input, axis=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )

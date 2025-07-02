from typing import Optional
from paddle_utils import add_tensor_methods
import paddle

__all__ = ["SoftBCEWithLogitsLoss"]

add_tensor_methods()
class SoftBCEWithLogitsLoss(paddle.nn.Layer):
    """
    Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight=None,
        ignore_index: Optional[int] = -100,
        reduction="mean",
        smooth_factor=None,
        pos_weight=None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer(name="weight", tensor=weight)
        self.register_buffer(name="pos_weight", tensor=pos_weight)

    def forward(self, input: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        if self.smooth_factor is not None:
            soft_targets = (
                (1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)
            ).astype(dtype=input.dtype)
        else:
            soft_targets = target.astype(dtype=input.dtype)
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit=input,
            label=soft_targets,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if self.ignore_index is not None:
            not_ignored_mask: paddle.Tensor = target != self.ignore_index
            loss *= not_ignored_mask.astype(dtype=loss.dtype)
        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss

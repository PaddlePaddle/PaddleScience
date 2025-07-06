from typing import Optional
from paddle_utils import add_tensor_methods
import paddle

__all__ = ["BalancedBCEWithLogitsLoss", "balanced_binary_cross_entropy_with_logits"]

add_tensor_methods()
def balanced_binary_cross_entropy_with_logits(
    logits: paddle.Tensor,
    targets: paddle.Tensor,
    gamma: float = 1.0,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> paddle.Tensor:
    """
    Balanced binary cross entropy loss.

    Args:
        logits:
        targets: This loss function expects target values to be hard targets 0/1.
        gamma: Power factor for balancing weights
        ignore_index:
        reduction:

    Returns:
        Zero-sized tensor with reduced loss if `reduction` is `sum` or `mean`; Otherwise returns loss of the
        shape of `logits` tensor.
    """
    pos_targets: paddle.Tensor = targets.equal(y=1).sum()
    neg_targets: paddle.Tensor = targets.equal(y=0).sum()
    num_targets = pos_targets + neg_targets
    pos_weight = paddle.pow(x=neg_targets / (num_targets + 1e-07), y=gamma)
    neg_weight = 1.0 - pos_weight
    pos_term = (
        pos_weight.pow(y=gamma) * targets * paddle.nn.functional.log_sigmoid(x=logits)
    )
    neg_term = (
        neg_weight.pow(y=gamma)
        * (1 - targets)
        * paddle.nn.functional.log_sigmoid(x=-logits)
    )
    loss = -(pos_term + neg_term)
    if ignore_index is not None:
        loss = paddle.masked_fill(x=loss, mask=targets.equal(y=ignore_index), value=0)
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    return loss


class BalancedBCEWithLogitsLoss(paddle.nn.Layer):
    """
    Balanced binary cross-entropy loss.

    https://arxiv.org/pdf/1504.06375.pdf (Formula 2)
    """

    __constants__ = ["gamma", "reduction", "ignore_index"]

    def __init__(
        self, gamma: float = 1.0, reduction="mean", ignore_index: Optional[int] = None
    ):
        """

        Args:
            gamma:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        return balanced_binary_cross_entropy_with_logits(
            output,
            target,
            gamma=self.gamma,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

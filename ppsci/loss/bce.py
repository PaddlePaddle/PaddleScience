from typing import Dict
from typing import Optional
from typing import Union

import paddle
import paddle.nn.functional as F
from typing_extensions import Literal

from ppsci.loss import base


class BCELoss(base.Loss):
    r"""Binary cross-entropy (BCE) loss with logits.

    Given logits tensor :math:`\mathbf{z}` and binary targets :math:`\mathbf{y}\in\{0,1\}`,
    the element-wise BCE (with logits) is
    \[
        \ell(\mathbf{z}, \mathbf{y}) =
        \text{BCEWithLogits}(\mathbf{z}, \mathbf{y})
        = \max(\mathbf{z}, 0) - \mathbf{z}\odot \mathbf{y} + \log(1 + e^{-|\mathbf{z}|})
    \]
    We then aggregate along the feature dimension (axis=1) and apply reduction.

    If `output_dict` contains key `"area"`, the per-element loss will be multiplied
    by `output_dict["area"]` before aggregation.

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Overall or per-key loss weight. Defaults to None.

    Inputs:
        output_dict (Dict[str, Tensor]): Must contain logits for each supervised key.
        label_dict  (Dict[str, Tensor]): Binary targets (0/1) with the same shape as logits.
        weight_dict (Optional[Dict[str, float]]): Optional per-key extra weights.

    Returns:
        Dict[str, paddle.Tensor]: A dict of reduced losses per key.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        super().__init__(reduction, weight)

    def forward(
        self, output_dict, label_dict, weight_dict=None
    ) -> Dict[str, "paddle.Tensor"]:
        losses: Dict[str, paddle.Tensor] = {}
        # print(label_dict)
        for key in label_dict:
            # logits and targets must have same shape
            logits = output_dict[key]
            targets = label_dict[key]

            # element-wise BCE with logits, no reduction
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

            # aggregate over feature dimension (axis=1) to get per-sample loss
            # (keep the same aggregation pattern as your L2Loss, but without sqrt)
            if loss.ndim >= 2:
                loss = loss.sum(axis=1)

            # reduction over batch
            if self.reduction == "sum":
                loss = loss.sum()
            else:  # "mean"
                loss = loss.mean()

            # final global/per-key weighting like in L2Loss

            losses[key] = loss
        print(losses)
        return losses


class FocalLoss(base.Loss):
    r"""Binary Focal Loss (with logits), suitable for class-imbalance.

    The Focal Loss is defined (per element) as:
    \[
        \text{FL} = \alpha_t (1 - p_t)^\gamma \cdot \text{BCEWithLogits}(\mathbf{z}, \mathbf{y}),
    \]
    where
    \[
        p = \sigma(\mathbf{z}),\quad
        p_t = p\mathbf{y} + (1-p)(1-\mathbf{y}),\quad
        \alpha_t = \alpha\mathbf{y} + (1-\alpha)(1-\mathbf{y}).
    \]

    We compute element-wise FL, allow optional `"area"` weighting, then sum over
    feature dimension (axis=1), and finally apply global `reduction`.

    Args:
        reduction (Literal["mean", "sum"], optional): Reduction method. Defaults to "mean".
        weight (Optional[Union[float, Dict[str, float]]]): Overall or per-key loss weight. Defaults to None.
        alpha (float, optional): Class balancing factor in [0,1]. Defaults to 0.25.
        gamma (float, optional): Focusing parameter (>=0). Defaults to 2.0.

    Inputs:
        output_dict (Dict[str, Tensor]): Must contain logits for each supervised key.
        label_dict  (Dict[str, Tensor]): Binary targets (0/1) with the same shape as logits.
        weight_dict (Optional[Dict[str, float]]): Optional per-key extra weights.

    Returns:
        Dict[str, paddle.Tensor]: A dict of reduced losses per key.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum"] = "mean",
        weight: Optional[Union[float, Dict[str, float]]] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction should be 'mean' or 'sum', but got {reduction}"
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], but got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, but got {gamma}")

        super().__init__(reduction, weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self, output_dict, label_dict, weight_dict=None
    ) -> Dict[str, "paddle.Tensor"]:
        losses: Dict[str, paddle.Tensor] = {}

        for key in label_dict:
            logits = output_dict[key]
            targets = label_dict[key]

            # base BCE with logits (element-wise, no reduction)
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

            # probabilities with sigmoid
            p = paddle.nn.functional.sigmoid(logits)
            # pt = p for y=1; (1-p) for y=0
            pt = p * targets + (1.0 - p) * (1.0 - targets)

            # alpha_t = alpha for y=1; (1-alpha) for y=0
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

            # focal modulation
            focal_factor = paddle.pow(1.0 - pt, self.gamma)

            loss = alpha_t * focal_factor * bce

            # optional area weighting
            if "area" in output_dict:
                loss = loss * output_dict["area"]

            # optional extra key-wise weight_dict
            if weight_dict and key in weight_dict:
                loss = loss * weight_dict[key]

            # aggregate over feature dimension (axis=1) to get per-sample loss
            if loss.ndim >= 2:
                loss = loss.sum(axis=1)

            # global reduction over batch
            if self.reduction == "sum":
                loss = loss.sum()
            else:  # "mean"
                loss = loss.mean()

            # final global/per-key weighting
            if isinstance(self.weight, (float, int)):
                loss = loss * float(self.weight)
            elif isinstance(self.weight, dict) and key in self.weight:
                loss = loss * self.weight[key]

            losses[key] = loss

        return losses

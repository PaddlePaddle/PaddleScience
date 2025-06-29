import logging
from typing import Optional

import paddle
import paddle.nn.functional as F

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"
EPS = 1e-10
logger = logging.getLogger(__name__)


def expand_onehot_labels(labels, target_shape, ignore_index):
    valid_mask = (labels >= 0) & (labels != ignore_index)
    num_classes = target_shape[1]
    if labels.dtype != paddle.int64 and labels.dtype != paddle.int32:
        labels = labels.astype('int64')
    safe_labels = paddle.where(valid_mask, labels, paddle.zeros_like(labels))
    bin_labels = paddle.nn.functional.one_hot(
        safe_labels, 
        num_classes=num_classes
    )
    bin_labels = bin_labels.transpose([0, 3, 1, 2])
    bin_labels = bin_labels * valid_mask.astype(bin_labels.dtype).unsqueeze(1)
    
    return bin_labels, valid_mask


def get_region_proportion(
    x: paddle.Tensor, valid_mask: paddle.Tensor = None
) -> paddle.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.astype(x.dtype)
            x = paddle.einsum("bcwh, bcwh->bcwh", x, valid_mask)
            cardinality = paddle.einsum("bcwh->bc", valid_mask)
        else:
            valid_mask = valid_mask.astype(x.dtype)
            x = paddle.einsum("bcwh,bwh->bcwh", x, valid_mask)
            cardinality = (
                paddle.einsum("bwh->b", valid_mask)
                .unsqueeze(axis=1)
                .tile(repeat_times=[1, tuple(x.shape)[1]])
            )
    else:
        cardinality = tuple(x.shape)[2] * tuple(x.shape)[3]
    region_proportion = (paddle.einsum("bcwh->bc", x) + EPS) / (cardinality + EPS)
    return region_proportion


class CompoundLoss(paddle.nn.Layer):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """

    def __init__(
        self,
        mode: str = MULTICLASS_MODE,
        alpha: float = 0.1,
        factor: float = 5.0,
        step_size: int = 0,
        max_alpha: float = 100.0,
        temp: float = 1.0,
        ignore_index: int = 255,
        background_index: int = -1,
        weight: Optional[paddle.Tensor] = None,
    ) -> None:
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: paddle.Tensor, labels: paddle.Tensor):
        if self.mode == MULTICLASS_MODE:
            if labels.dtype != paddle.int64 and labels.dtype != paddle.int32:
                labels = labels.astype('int64')
        
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            loss = F.cross_entropy(
                input=inputs,
                label=labels,
                weight=self.weight,
                ignore_index=self.ignore_index,
                axis=1  
                )
        else:
            if self.mode == BINARY_MODE:
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(
                        inputs, 
                        labels.astype('float32')
                        )
            else: 
                loss = F.binary_cross_entropy_with_logits(
                    inputs, 
                    labels.astype('float32')
                    )
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            logger.info(
                "CompoundLoss : Adjust the tradoff param alpha : {:.3g} -> {:.3g}".format(
                    curr_alpha, self.alpha
                )
            )

    def get_gt_proportion(
        self, mode: str, labels: paddle.Tensor, target_shape, ignore_index: int = 255
    ):
        if mode == MULTICLASS_MODE:
            bin_labels, valid_mask = expand_onehot_labels(
                labels, target_shape, ignore_index
            )
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            if labels.dim() == 3:
                labels = labels.unsqueeze(axis=1)
            bin_labels = labels
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(
        self, mode: str, logits: paddle.Tensor, temp: float = 1.0, valid_mask=None
    ):
        if mode == MULTICLASS_MODE:
            preds = F.log_softmax(x=temp * logits, axis=1).exp()
        else:
            preds = F.log_sigmoid(x=temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion


class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """

    def forward(self, inputs: paddle.Tensor, labels: paddle.Tensor):
        loss_ce = self.cross_entropy(inputs, labels)
        gt_proportion, valid_mask = self.get_gt_proportion(
            self.mode, labels, tuple(inputs.shape)
        )
        pred_proportion = self.get_pred_proportion(
            self.mode, inputs, temp=self.temp, valid_mask=valid_mask
        )
        loss_reg = (pred_proportion - gt_proportion).abs().mean()
        loss = loss_ce + self.alpha * loss_reg
        return loss


class CrossEntropyWithKL(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * KL(gt_region || prob_region)
    """

    def kl_div(self, p: paddle.Tensor, q: paddle.Tensor) -> paddle.Tensor:
        x = p * paddle.log(x=p / q)
        x = paddle.einsum("ij->i", x)
        return x

    def forward(self, inputs: paddle.Tensor, labels: paddle.Tensor):
        loss_ce = self.cross_entropy(inputs, labels)
        gt_proportion, valid_mask = self.get_gt_proportion(
            self.mode, labels, tuple(inputs.shape)
        )
        pred_proportion = self.get_pred_proportion(
            self.mode, inputs, temp=self.temp, valid_mask=valid_mask
        )
        if self.mode == BINARY_MODE:
            regularizer = (
                self.kl_div(gt_proportion, pred_proportion)
                + self.kl_div(1 - gt_proportion, 1 - pred_proportion)
            ).mean()
        else:
            regularizer = self.kl_div(gt_proportion, pred_proportion).mean()
        loss = loss_ce + self.alpha * regularizer
        return loss

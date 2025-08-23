import paddle.nn as nn

from ppsci.registry import register


@register
class CrossEntropyLoss(nn.Layer):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

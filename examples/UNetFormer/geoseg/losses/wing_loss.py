import paddle

from . import functional as F

__all__ = ["WingLoss"]

class WingLoss(paddle.nn.Layer):
    def __init__(self, width=5, curvature=0.5, reduction="mean"):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.reduction = reduction

    def forward(self, prediction, target):
        return F.wing_loss(
            prediction, target, self.width, self.curvature, self.reduction
        )

"""
Converted to PaddlePaddle
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class FocalLoss(nn.Layer):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_label=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = paddle.to_tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = paddle.to_tensor(alpha)
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, input, target):
        if input.ndim > 2:
            input = input.reshape(
                [input.shape[0], input.shape[1], -1]
            )  # N,C,H,W => N,C,H*W
            input = input.transpose([0, 2, 1])  # N,C,H*W => N,H*W,C
            input = input.reshape([-1, input.shape[2]])  # N,H*W,C => N*H*W,C
        target = target.reshape([-1, 1])

        if input.squeeze(1).ndim == 1:
            logpt = F.sigmoid(input)
            logpt = logpt.flatten()
        else:
            logpt = F.log_softmax(input, axis=-1)
            logpt = paddle.gather_nd(
                logpt,
                paddle.stack([paddle.arange(logpt.shape[0]), target.squeeze()], axis=1),
            )
            logpt = logpt.flatten()

        pt = paddle.exp(logpt)

        if self.alpha is not None:
            if self.alpha.dtype != input.dtype:
                self.alpha = self.alpha.astype(input.dtype)
            at = paddle.gather(self.alpha, target.flatten().astype("int64"))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.ignore_label is not None:
            valid_mask = target[:, 0] != self.ignore_label
            loss = loss[valid_mask]

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

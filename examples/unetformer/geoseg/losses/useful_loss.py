import paddle
from paddle_utils import add_tensor_methods
from paddle_utils import device2int

from .dice import DiceLoss
from .joint_loss import JointLoss
from .soft_ce import SoftCrossEntropyLoss

add_tensor_methods()
__all__ = ["EdgeLoss", "OHEM_CELoss", "UnetFormerLoss"]


class EdgeLoss(paddle.nn.Layer):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        out_0 = paddle.to_tensor(
            data=[-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype="float32"
        ).reshape(1, 1, 3, 3)
        out_0.stop_gradient = not False
        laplacian_kernel_target = out_0.cuda(device_id=device2int(x.place))
        x = x.unsqueeze(axis=1).astype(dtype="float32")
        x = paddle.nn.functional.conv2d(x=x, weight=laplacian_kernel_target, padding=1)
        x = x.clip(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0
        return x

    def compute_edge_loss(self, logits, targets):
        bs = tuple(logits.shape)[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        logits = (
            paddle.nn.functional.softmax(x=logits, axis=1)
            .argmax(axis=1)
            .squeeze(axis=1)
        )
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        edge_loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit=boundary_pre, label=boundary_targets
        )
        return edge_loss

    def forward(self, logits, targets):
        loss = (
            self.main_loss(logits, targets)
            + self.compute_edge_loss(logits, targets) * self.edge_factor
        ) / (self.edge_factor + 1)
        return loss


class OHEM_CELoss(paddle.nn.Layer):
    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -paddle.log(
            x=paddle.to_tensor(data=thresh, dtype="float32", stop_gradient=not False)
        ).cuda()
        self.ignore_index = ignore_index
        self.criteria = paddle.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits, labels):
        if logits.shape[2:] != labels.shape[1:]:
            logits = paddle.nn.interpolate(
                logits, size=labels.shape[1:], mode="bilinear", align_corners=True
            )
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("Batch size mismatch between logits and labels")
        logits = logits.transpose([0, 2, 3, 1])
        logits = logits.reshape([-1, logits.shape[3]])
        labels = labels.reshape([-1])
        valid_mask = labels != self.ignore_index
        n_valid = paddle.sum(valid_mask).item()
        n_min = max(1, n_valid // 16)
        loss = self.criteria(logits, labels)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.size < n_min:
            loss_hard, _ = loss.topk(k=n_min)
        return paddle.mean(x=loss_hard)


class UnetFormerLoss(paddle.nn.Layer):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index),
            1.0,
            1.0,
        )
        self.aux_loss = SoftCrossEntropyLoss(
            smooth_factor=0.05, ignore_index=ignore_index
        )

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(
                logit_aux, labels
            )
        else:
            loss = self.main_loss(logits, labels)
        return loss


if __name__ == "__main__":
    targets = paddle.randint(low=0, high=2, shape=(2, 16, 16))
    logits = paddle.randn(shape=(2, 2, 16, 16))
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)
    print(loss)

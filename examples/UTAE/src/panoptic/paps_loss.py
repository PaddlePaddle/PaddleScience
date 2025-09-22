"""
PaPs Implementation (Paddle Version)
Converted to PaddlePaddle
"""

import paddle
import paddle.nn as nn
from src.panoptic.FocalLoss import FocalLoss


class PaPsLoss(nn.Layer):
    """
    Loss for training PaPs.
    Args:
        l_center (float): Coefficient for the centerness loss (default 1)
        l_size (float): Coefficient for the size loss (default 1)
        l_shape (float): Coefficient for the shape loss (default 1)
        l_class (float): Coefficient for the classification loss (default 1)
        alpha (float): Parameter for the centerness loss (default 0)
        beta (float): Parameter for the centerness loss (default 4)
        gamma (float): Focal exponent for the classification loss (default 0)
        eps (float): Stability epsilon
        void_label (int): Label to ignore in the classification loss
    """

    def __init__(
        self,
        l_center=1,
        l_size=1,
        l_shape=1,
        l_class=1,
        alpha=0,
        beta=4,
        gamma=0,
        eps=1e-8,
        void_label=None,
        binary_threshold=0.4,
    ):

        super(PaPsLoss, self).__init__()
        self.l_center = l_center
        self.l_size = l_size
        self.l_shape = l_shape
        self.l_class = l_class
        self.eps = eps
        self.binary_threshold = binary_threshold

        self.center_loss = CenterLoss(alpha=alpha, beta=beta, eps=eps)
        self.class_loss = FocalLoss(gamma=gamma, ignore_label=void_label)
        self.shape_loss = FocalLoss(gamma=0)
        self.value = (0, 0, 0, 0, 0)

        # Keep track of the predicted confidences and ious between predicted and gt binary masks.
        # This is usefull for tuning the confidence threshold of the pseudo-nms.
        self.predicted_confidences = None
        self.achieved_ious = None

    def forward(self, predictions, target, heatmap_only=False):
        # Split target tensor - equivalent to torch.split
        target_splits = paddle.split(target, [1, 1, 1, 2, 1, 1], axis=-1)
        (
            target_heatmap,
            true_instances,
            zones,
            size,
            sem_obj,
            sem_pix,
        ) = target_splits

        # Create center mapping dictionary - following original torch logic
        # center_mask is now always 3D (B, H, W) from PaPs model
        center_mask = predictions["center_mask"]
        center_indices = paddle.nonzero(center_mask)
        center_mapping = {}

        if center_indices.shape[0] > 0:
            # Create mapping: (batch_idx, height_idx, width_idx) -> center_id
            for k, (b, i, j) in enumerate(
                zip(center_indices[:, 0], center_indices[:, 1], center_indices[:, 2])
            ):
                center_mapping[(int(b), int(i), int(j))] = k

        loss_center = 0
        loss_size = 0
        loss_shape = 0
        loss_class = 0

        if self.l_center != 0:
            loss_center = self.center_loss(predictions["heatmap"], target_heatmap)

        if not heatmap_only and predictions["size"].shape[0] != 0:
            if self.l_size != 0:
                # Use center indices to extract corresponding sizes
                if center_indices.shape[0] > 0:
                    # Extract true sizes at center locations
                    # center_indices now has shape (N, 3) with (batch, height, width)
                    batch_ids = center_indices[:, 0]
                    h_ids = center_indices[:, 1]
                    w_ids = center_indices[:, 2]

                    # Use gather_nd to extract sizes
                    size_indices = paddle.stack([batch_ids, h_ids, w_ids], axis=1)
                    true_size = paddle.gather_nd(size, size_indices)  # (N, 2)

                    loss_size = paddle.abs(true_size - predictions["size"]) / (
                        true_size + self.eps
                    )
                    loss_size = loss_size.sum(axis=-1).mean()
                else:
                    loss_size = paddle.to_tensor(0.0)

            if self.l_class != 0:
                # Use center indices for semantic object labels
                if center_indices.shape[0] > 0:
                    # Extract semantic labels at center locations
                    # center_indices now has shape (N, 3) with (batch, height, width)
                    batch_ids = center_indices[:, 0]
                    h_ids = center_indices[:, 1]
                    w_ids = center_indices[:, 2]

                    # Use gather_nd to extract semantic labels
                    sem_indices = paddle.stack([batch_ids, h_ids, w_ids], axis=1)
                    sem_labels = paddle.gather_nd(sem_obj, sem_indices)  # (N, 1)
                    sem_labels = sem_labels.squeeze(
                        -1
                    )  # Remove last dimension to get (N,)

                    loss_class = self.class_loss(
                        predictions["semantic"],
                        sem_labels.astype("int64"),
                    )
                else:
                    loss_class = paddle.to_tensor(0.0)

            if self.l_shape != 0:
                confidence_pred = []
                ious = []
                flatten_preds = []
                flatten_target = []

                # Faithful to original PyTorch implementation
                for b in range(true_instances.shape[0]):
                    instance_mask = true_instances[b]
                    for inst_id in paddle.unique(instance_mask):
                        centers = predictions["center_mask"][b] * (
                            zones[b] == inst_id
                        ).squeeze(
                            -1
                        )  # center matching

                        if not centers.any():
                            continue

                        # Original PyTorch style: iterate over nonzero positions
                        center_positions = paddle.nonzero(centers)
                        for pos in center_positions:
                            x, y = int(pos[0]), int(pos[1])
                            true_mask = (
                                (instance_mask == inst_id).squeeze(-1).astype("float32")
                            )

                            pred_id = center_mapping[(b, int(x), int(y))]

                            xtl, ytl, xbr, ybr = predictions["instance_boxes"][pred_id]

                            crop_true = true_mask[ytl:ybr, xtl:xbr].reshape([-1, 1])
                            mask = predictions["instance_masks"][pred_id].reshape(
                                [-1, 1]
                            )

                            flatten_preds.append(mask)
                            flatten_target.append(crop_true)

                            confidence_pred.append(predictions["confidence"][pred_id])
                            bmask = (mask > self.binary_threshold).astype("float32")
                            inter = (bmask * crop_true).sum().astype("float32")
                            union = ((bmask + crop_true) != 0).astype("float32").sum()
                            true_mask[ytl:ybr, xtl:xbr] = 0
                            union = (
                                union + true_mask.sum()
                            )  # parts of shape outside of bbox
                            iou = inter / union
                            if paddle.isnan(iou) or paddle.isinf(iou):
                                iou = paddle.zeros([1], dtype="float32")
                            ious.append(iou)

                if len(flatten_preds) > 0:
                    p = paddle.concat(flatten_preds, axis=0)
                    p = paddle.concat([1 - p, p], axis=1)
                    t = paddle.concat(flatten_target, axis=0).astype("int64")
                    loss_shape = self.shape_loss(p, t)

                    self.predicted_confidences = paddle.stack(confidence_pred)
                    self.achieved_ious = paddle.stack(ious).unsqueeze(-1)
                else:
                    loss_shape = paddle.to_tensor(0.0)

        loss = (
            self.l_center * loss_center
            + self.l_size * loss_size
            + self.l_shape * loss_shape
            + self.l_class * loss_class
        )

        self.value = (
            float(loss_center.detach().cpu().item())
            if isinstance(loss_center, paddle.Tensor)
            else loss_center,
            float(loss_size.detach().cpu().item())
            if isinstance(loss_size, paddle.Tensor)
            else loss_size,
            float(loss_shape.detach().cpu().item())
            if isinstance(loss_shape, paddle.Tensor)
            else loss_shape,
            float(loss_class.detach().cpu().item())
            if isinstance(loss_class, paddle.Tensor)
            else loss_class,
        )
        return loss


class CenterLoss(nn.Layer):
    """
    Adapted from the github repo of the CornerNet paper
    https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py
    Converted to PaddlePaddle
    """

    def __init__(self, alpha=0, beta=4, eps=1e-8):
        super(CenterLoss, self).__init__()
        self.a = alpha
        self.b = beta
        self.eps = eps

    def forward(self, preds, gt):
        pred = preds.transpose([0, 2, 3, 1]).reshape([-1, preds.shape[1]])
        g = gt.reshape([-1, preds.shape[1]])

        pos_inds = g == 1
        neg_inds = g < 1
        num_pos = pos_inds.astype("float32").sum()
        loss = 0

        if pos_inds.any():
            pos_pred = pred[pos_inds]
            pos_loss = paddle.log(pos_pred + self.eps)
            pos_loss = pos_loss * paddle.pow(1 - pos_pred, self.a)
            pos_loss = pos_loss.sum()
        else:
            pos_loss = paddle.to_tensor(0.0)

        if neg_inds.any():
            neg_pred = pred[neg_inds]
            neg_g = g[neg_inds]
            neg_loss = paddle.log(1 - neg_pred + self.eps)
            neg_loss = neg_loss * paddle.pow(neg_pred, self.a)
            neg_loss = neg_loss * paddle.pow(1 - neg_g, self.b)
            neg_loss = neg_loss.sum()
        else:
            neg_loss = paddle.to_tensor(0.0)

        if not pos_inds.any():
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

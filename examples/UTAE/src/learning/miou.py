"""
IoU metric computation (Paddle Version)
"""
import numpy as np
import paddle


class IoU:
    def __init__(self, num_classes, ignore_index=-1, cm_device="cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm_device = cm_device
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    """
    Add predictions and targets to confusion matrix
    """

    def add(self, pred, target):
        # Convert to numpy if tensors
        if isinstance(pred, paddle.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, paddle.Tensor):
            target = target.cpu().numpy()

        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()

        # Remove ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]

        # Compute confusion matrix
        for t, p in zip(target.flatten(), pred.flatten()):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    """
    Get mean IoU and accuracy from confusion matrix
    """

    def get_miou_acc(self):

        # Overall accuracy
        acc = np.diag(self.confusion_matrix).sum() / (
            self.confusion_matrix.sum() + 1e-15
        )

        # Per-class IoU
        ious = []
        for i in range(self.num_classes):
            intersection = self.confusion_matrix[i, i]
            union = (
                self.confusion_matrix[i, :].sum()
                + self.confusion_matrix[:, i].sum()
                - intersection
            )

            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(0.0)

        # Mean IoU
        miou = np.mean(ious)

        return miou, acc

    """
    Reset confusion matrix
    """

    def reset(self):

        self.confusion_matrix.fill(0)

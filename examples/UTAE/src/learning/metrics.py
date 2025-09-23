"""
Metrics utilities (Paddle Version)
"""
import numpy as np

"""
Compute per-class and overall metrics from confusion matrix
"""


def confusion_matrix_analysis(cm):

    n_classes = cm.shape[0]

    # Overall accuracy
    acc = np.diag(cm).sum() / (cm.sum() + 1e-15)

    # Per-class metrics
    per_class_metrics = {}
    ious = []

    for i in range(n_classes):
        # True positives, false positives, false negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        # Precision, recall, F1
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)

        # IoU
        union = tp + fp + fn
        iou = tp / (union + 1e-15)
        ious.append(iou)

        per_class_metrics[f"class_{i}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    # Mean metrics
    mean_iou = np.mean(ious)

    return {
        "overall_accuracy": acc,
        "mean_iou": mean_iou,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }

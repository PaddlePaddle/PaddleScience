"""
Panoptic Metrics (Paddle Version)
Converted to PaddlePaddle
"""

import paddle


class PanopticMeter:
    """
    Meter class for the panoptic metrics as defined by Kirilov et al. :
    Segmentation Quality (SQ)
    Recognition Quality (RQ)
    Panoptic Quality (PQ)
    The behavior of this meter mimics that of torchnet meters, each predicted batch
    is added via the add method and the global metrics are retrieved with the value
    method.
    Args:
        num_classes (int): Number of semantic classes (including background and void class).
        void_label (int): Label for the void class (default 19).
        background_label (int): Label for the background class (default 0).
        iou_threshold (float): Threshold used on the IoU of the true vs predicted
        instance mask. Above the threshold a true instance is counted as True Positive.
    """

    def __init__(
        self, num_classes=20, background_label=0, void_label=19, iou_threshold=0.5
    ):

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.class_list = [c for c in range(num_classes) if c != background_label]
        self.void_label = void_label
        if void_label is not None:
            self.class_list = [c for c in self.class_list if c != void_label]
        self.counts = paddle.zeros([len(self.class_list), 3])
        self.cumulative_ious = paddle.zeros([len(self.class_list)])

    def add(self, predictions, target):
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

        instance_true = true_instances.squeeze(-1)
        semantic_true = sem_pix.squeeze(-1)

        instance_pred = predictions["pano_instance"]

        # Handle case when pano_semantic is None (when pseudo_nms=False)
        if predictions["pano_semantic"] is not None:
            semantic_pred = predictions["pano_semantic"].argmax(axis=1)
        else:
            # Return early with zero metrics when no panoptic predictions available
            return

        if self.void_label is not None:
            void_masks = (semantic_true == self.void_label).astype("float32")

            # Ignore Void Objects
            for batch_idx in range(void_masks.shape[0]):
                void_mask = void_masks[batch_idx]
                if void_mask.any():
                    void_instances = instance_true[batch_idx] * void_mask
                    unique_void, void_counts = paddle.unique(
                        void_instances, return_counts=True
                    )

                    for void_inst_id, void_inst_area in zip(unique_void, void_counts):
                        if void_inst_id == 0:
                            continue

                        pred_instances = instance_pred[batch_idx]
                        unique_pred, pred_counts = paddle.unique(
                            pred_instances, return_counts=True
                        )

                        for pred_inst_id, pred_inst_area in zip(
                            unique_pred, pred_counts
                        ):
                            if pred_inst_id == 0:
                                continue
                            inter = (
                                (instance_true[batch_idx] == void_inst_id)
                                * (instance_pred[batch_idx] == pred_inst_id)
                            ).sum()
                            iou = inter.astype("float32") / (
                                void_inst_area + pred_inst_area - inter
                            ).astype("float32")
                            if iou > self.iou_threshold:
                                instance_pred[batch_idx] = paddle.where(
                                    instance_pred[batch_idx] == pred_inst_id,
                                    paddle.to_tensor(0),
                                    instance_pred[batch_idx],
                                )
                                semantic_pred[batch_idx] = paddle.where(
                                    instance_pred[batch_idx] == pred_inst_id,
                                    paddle.to_tensor(0),
                                    semantic_pred[batch_idx],
                                )

            # Ignore Void Pixels
            instance_pred = paddle.where(void_masks, paddle.to_tensor(0), instance_pred)
            semantic_pred = paddle.where(void_masks, paddle.to_tensor(0), semantic_pred)

        # Compute metrics for each class
        for i, class_id in enumerate(self.class_list):
            TP = 0
            n_preds = 0
            n_true = 0
            ious = []

            for batch_idx in range(instance_true.shape[0]):
                instance_mask = instance_true[batch_idx]
                class_mask_gt = (semantic_true[batch_idx] == class_id).astype("float32")
                class_mask_p = (semantic_pred[batch_idx] == class_id).astype("float32")

                pred_class_instances = instance_pred[batch_idx] * class_mask_p
                true_class_instances = instance_mask * class_mask_gt

                n_preds += (
                    int(paddle.unique(pred_class_instances).shape[0]) - 1
                )  # do not count 0
                n_true += int(paddle.unique(true_class_instances).shape[0]) - 1

                if n_preds == 0 or n_true == 0:
                    continue  # no true positives in that case

                unique_true, true_counts = paddle.unique(
                    true_class_instances, return_counts=True
                )
                for true_inst_id, true_inst_area in zip(unique_true, true_counts):
                    if true_inst_id == 0:  # masked segments
                        continue

                    unique_pred, pred_counts = paddle.unique(
                        pred_class_instances, return_counts=True
                    )
                    for pred_inst_id, pred_inst_area in zip(unique_pred, pred_counts):
                        if pred_inst_id == 0:
                            continue
                        inter = (
                            (instance_mask == true_inst_id)
                            * (instance_pred[batch_idx] == pred_inst_id)
                        ).sum()
                        iou = inter.astype("float32") / (
                            true_inst_area + pred_inst_area - inter
                        ).astype("float32")

                        if iou > self.iou_threshold:
                            TP += 1
                            ious.append(iou)

            FP = n_preds - TP
            FN = n_true - TP

            self.counts[i] += paddle.to_tensor([TP, FP, FN], dtype="float32")
            if len(ious) > 0:
                self.cumulative_ious[i] += paddle.stack(ious).sum()

    def value(self, per_class=False):
        TP, FP, FN = paddle.split(self.counts.astype("float32"), 3, axis=-1)
        SQ = self.cumulative_ious / TP.squeeze()

        # Handle NaN and Inf values
        nan_mask = paddle.isnan(SQ) | paddle.isinf(SQ)
        SQ = paddle.where(nan_mask, paddle.to_tensor(0.0), SQ)

        RQ = TP / (TP + 0.5 * FP + 0.5 * FN)
        PQ = SQ * RQ.squeeze(-1)

        if per_class:
            return SQ, RQ, PQ
        else:
            valid_mask = ~paddle.isnan(PQ)
            if valid_mask.any():
                return (
                    SQ[valid_mask].mean(),
                    RQ[valid_mask].mean(),
                    PQ[valid_mask].mean(),
                )
            else:
                return (
                    paddle.to_tensor(0.0),
                    paddle.to_tensor(0.0),
                    paddle.to_tensor(0.0),
                )

    def get_table(self):
        table = (
            paddle.concat([self.counts, self.cumulative_ious[:, None]], axis=-1)
            .cpu()
            .numpy()
        )
        return table

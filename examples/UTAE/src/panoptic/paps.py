"""
PaPs Implementation (Paddle Version)
Converted to PaddlePaddle
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.backbones.utae import ConvLayer


class PaPs(nn.Layer):
    """
    Implementation of the Parcel-as-Points Module (PaPs) for panoptic segmentation of agricultural
    parcels from satellite image time series.
    Args:
        encoder (nn.Layer): Backbone encoding network. The encoder is expected to return
        a feature map at the same resolution as the input images and a list of feature maps
        of lower resolution.
        num_classes (int): Number of classes (including stuff and void classes).
        shape_size (int): S hyperparameter defining the shape of the local patch.
        mask_conv (bool): If False no residual CNN is applied after combination of
        the predicted shape and the cropped saliency (default True)
        min_confidence (float): Cut-off confidence level for the pseudo NMS (predicted instances with
        lower condidence will not be included in the panoptic prediction).
        min_remain (float): Hyperparameter of the pseudo-NMS that defines the fraction of a candidate instance mask
        that needs to be new to be included in the final panoptic prediction (default  0.5).
        mask_threshold (float): Binary threshold for instance masks (default 0.4)
    """

    def __init__(
        self,
        encoder,
        num_classes=20,
        shape_size=16,
        mask_conv=True,
        min_confidence=0.2,
        min_remain=0.5,
        mask_threshold=0.4,
    ):

        super(PaPs, self).__init__()
        self.encoder = encoder
        self.shape_size = shape_size
        self.num_classes = num_classes
        self.min_scale = 1 / shape_size
        self.register_buffer("min_confidence", paddle.to_tensor([min_confidence]))
        self.min_remain = min_remain
        self.mask_threshold = mask_threshold
        self.center_extractor = CenterExtractor()

        enc_dim = encoder.enc_dim
        stack_dim = encoder.stack_dim
        self.heatmap_conv = nn.Sequential(
            ConvLayer(
                nkernels=[enc_dim, 32, 1],
                last_relu=False,
                k=3,
                p=1,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )

        self.saliency_conv = ConvLayer(
            nkernels=[enc_dim, 32, 1], last_relu=False, k=3, p=1, padding_mode="reflect"
        )

        self.shape_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1D(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, shape_size**2),
        )

        self.size_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1D(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.BatchNorm1D(stack_dim // 4),
            nn.ReLU(),
            nn.Linear(stack_dim // 4, 2),
            nn.Softplus(),
        )

        self.class_mlp = nn.Sequential(
            nn.Linear(stack_dim, stack_dim // 2),
            nn.BatchNorm1D(stack_dim // 2),
            nn.ReLU(),
            nn.Linear(stack_dim // 2, stack_dim // 4),
            nn.Linear(stack_dim // 4, num_classes),
        )

        if mask_conv:
            self.mask_cnn = nn.Sequential(
                nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.GroupNorm(num_channels=16, num_groups=1),
                nn.ReLU(),
                nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2D(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            )
        else:
            self.mask_cnn = None

    def forward(
        self,
        input,
        batch_positions=None,
        zones=None,
        pseudo_nms=True,
        heatmap_only=False,
    ):
        """
        Args:
            input (tensor): Input image time series.
            batch_positions (tensor): Date sequence of the batch images.
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection). This mapping
            is used at train time to predict and supervise at most one prediction
            per ground truth object for efficiency.
            When not provided all predicted centers receive supervision.
            pseudo_nms (bool): If True performs pseudo_nms to produce a panoptic prediction,
            otherwise the model returns potentially overlapping instance segmentation masks (default True).
            heatmap_only (bool): If True the model only returns the centerness heatmap. Can be useful for some
            warmup epochs of the centerness prediction, as all the rest hinges on this.

        Returns:
            predictions (dict[tensor]): A dictionary of predictions with the following keys:
                center_mask         (B,H,W) Binary mask of centers.
                saliency            (B,1,H,W) Global Saliency.
                heatmap             (B,1,H,W) Predicted centerness heatmap.
                semantic            (M, K) Predicted class scores for each center (with M the number of predicted centers).
                size                (M, 2) Predicted sizes for each center.
                confidence          (M,1) Predicted centerness for each center.
                centerness          (M,1) Predicted centerness for each center.
                instance_masks      List of N binary masks of varying shape.
                instance_boxes      (N, 4) Coordinates of the N bounding boxes.
                pano_instance       (B,H,W) Predicted instance id for each pixel.
                pano_semantic       (B,K,H,W) Predicted class score for each pixel.

        """
        out, maps = self.encoder(input, batch_positions=batch_positions)

        # Global Predictions
        heatmap = self.heatmap_conv(out)
        saliency = self.saliency_conv(out)

        center_mask, _ = self.center_extractor(
            heatmap, zones=zones
        )  # (B,H,W) mask of N detected centers
        # Don't squeeze batch dimension to maintain consistency with loss function expectations

        if heatmap_only:
            predictions = dict(
                center_mask=center_mask,
                saliency=None,
                heatmap=heatmap,
                semantic=None,
                size=None,
                offsets=None,
                confidence=None,
                instance_masks=None,
                instance_boxes=None,
                pano_instance=None,
                pano_semantic=None,
            )
            return predictions

        # Retrieve info of detected centers
        H, W = heatmap.shape[-2:]
        # center_mask is now always 3D (B, H, W)
        center_indices = paddle.nonzero(center_mask, as_tuple=False)

        if center_indices.shape[0] == 0:
            # Handle case where no centers detected
            center_batch = paddle.empty([0], dtype="int64")
            center_h = paddle.empty([0], dtype="int64")
            center_w = paddle.empty([0], dtype="int64")
            center_positions = paddle.empty([0, 2], dtype="int64")
        else:
            # center_mask is (B, H, W), so indices are (N, 3)
            center_batch = center_indices[:, 0]
            center_h = center_indices[:, 1]
            center_w = center_indices[:, 2]
            center_positions = paddle.stack([center_h, center_w], axis=1)

        # Construct multi-level feature stack for centers
        stack = []
        for i, m in enumerate(maps):
            h_mask = center_h // (2 ** (len(maps) - 1 - i))
            # Assumes resolution is divided by 2 at each level
            w_mask = center_w // (2 ** (len(maps) - 1 - i))
            m = m.transpose([0, 2, 3, 1])
            # Use paddle.gather_nd for advanced indexing
            indices = paddle.stack([center_batch, h_mask, w_mask], axis=1)
            stack.append(paddle.gather_nd(m, indices))
        stack = paddle.concat(stack, axis=1)

        # Center-level predictions
        size = self.size_mlp(stack)
        sem = self.class_mlp(stack)
        shapes = self.shape_mlp(stack)
        shapes = shapes.reshape([-1, 1, self.shape_size, self.shape_size])
        # (N,1,S,S) instance shapes

        # Extract centerness from heatmap at center positions
        # Use gather_nd to extract values at specific positions
        if center_h.shape[0] > 0:
            # Create indices for gather_nd: [batch_idx, channel_idx, h_idx, w_idx]
            batch_indices = center_batch.unsqueeze(1)  # [N, 1]
            channel_indices = paddle.zeros_like(batch_indices)  # [N, 1] - channel 0
            h_indices = center_h.unsqueeze(1)  # [N, 1]
            w_indices = center_w.unsqueeze(1)  # [N, 1]
            gather_indices = paddle.concat(
                [batch_indices, channel_indices, h_indices, w_indices], axis=1
            )
            centerness = paddle.gather_nd(heatmap, gather_indices).unsqueeze(-1)
        else:
            centerness = paddle.empty([0, 1])
        confidence = centerness

        # Instance Boxes Assembling
        ## Minimal box size of 1px
        ## Combine clamped sizes and center positions to obtain box coordinates
        clamp_size = size.detach().round().astype("int64").clip(min=1)
        half_size = clamp_size // 2
        remainder_size = clamp_size % 2
        start_hw = center_positions - half_size
        stop_hw = center_positions + half_size + remainder_size

        instance_boxes = paddle.concat([start_hw, stop_hw], axis=1)
        instance_boxes = paddle.clip(instance_boxes, min=0, max=H)
        instance_boxes = instance_boxes[:, [1, 0, 3, 2]]  # h,w,h,w to x,y,x,y

        valid_start = paddle.clip(
            -start_hw, min=0
        )  # if h=-5 crop the shape mask before the 5th pixel
        valid_stop = (stop_hw - start_hw) - paddle.clip(
            stop_hw - H, min=0
        )  # crop if h_stop > H

        # Instance Masks Assembling
        instance_masks = []
        # Manual splitting to match PyTorch behavior exactly
        # PyTorch: shapes.split(1, dim=0) gives list of [1, 1, S, S] tensors
        # PaddlePaddle: paddle.split() behaves differently, use manual approach
        for i in range(shapes.shape[0]):
            s = shapes[i : i + 1]  # [1, 1, S, S] - exactly like PyTorch split
            h, w = clamp_size[i]  # Box size
            w_start, h_start, w_stop, h_stop = instance_boxes[
                i
            ]  # Box coordinates (x,y,x,y format)
            h_start_valid, w_start_valid = valid_start[i]  # Part of the Box that lies
            h_stop_valid, w_stop_valid = valid_stop[i]  # within the image's extent

            ## Resample local shape mask - match PyTorch exactly
            # s is single shape [1, 1, shape_size, shape_size] from split
            pred_mask = F.interpolate(s, size=[h.item(), w.item()], mode="bilinear")
            pred_mask = pred_mask.squeeze(0)  # Remove batch dim -> [1, h, w]
            pred_mask = pred_mask[
                :, h_start_valid:h_stop_valid, w_start_valid:w_stop_valid
            ]

            ## Crop saliency
            batch_idx = int(center_batch[i])  # Ensure scalar index
            crop_saliency = saliency[batch_idx, :, h_start:h_stop, w_start:w_stop]

            ## Combine both
            if self.mask_cnn is None:
                pred_mask = F.sigmoid(pred_mask) * F.sigmoid(crop_saliency)
                # Debug: print shape for mask_cnn is None case (only if needed)
                # print(f"Debug - pred_mask shape (no mask_cnn): {pred_mask.shape}")
            else:
                pred_mask = pred_mask + crop_saliency
                # Ensure pred_mask is [C, H, W] before mask_cnn
                if pred_mask.ndim != 3:
                    raise ValueError(
                        f"pred_mask should be 3D [C,H,W], got shape {pred_mask.shape}"
                    )
                pred_mask = F.sigmoid(pred_mask) * F.sigmoid(
                    self.mask_cnn(pred_mask.unsqueeze(0)).squeeze(0)
                )

            # Debug: print shape when appending to instance_masks (only if needed)
            # print(f"Debug - appending pred_mask with shape: {pred_mask.shape}")
            instance_masks.append(pred_mask)

        # PSEUDO-NMS
        if pseudo_nms:
            panoptic_instance = []
            panoptic_semantic = []
            for b in range(center_mask.shape[0]):  # iterate over elements of batch
                panoptic_mask = paddle.zeros(center_mask[0].shape, dtype="float32")
                semantic_mask = paddle.zeros(
                    [self.num_classes] + list(center_mask[0].shape), dtype="float32"
                )

                # Get indices of centers in this batch element - match PyTorch exactly
                candidates = paddle.nonzero(center_batch == b).squeeze(-1)
                if candidates.ndim == 0:  # Handle single candidate case
                    candidates = candidates.unsqueeze(0)

                if len(candidates) > 0:
                    # Sort by confidence descending - match PyTorch logic exactly
                    candidate_confidences = confidence[candidates].squeeze(-1)
                    if candidate_confidences.ndim == 0:  # Handle single confidence case
                        candidate_confidences = candidate_confidences.unsqueeze(0)

                    # Use argsort to get indices, then get sorted values - match torch.sort behavior
                    sorted_indices = paddle.argsort(
                        candidate_confidences, descending=True
                    )
                    sorted_values = candidate_confidences[sorted_indices]

                    for n, (c, idx_in_candidates) in enumerate(
                        zip(sorted_values, sorted_indices)
                    ):
                        if c < self.min_confidence:
                            break
                        else:
                            # Get the actual index in the original candidates array
                            actual_idx = candidates[idx_in_candidates]

                            new_mask = paddle.zeros(
                                center_mask[0].shape, dtype="float32"
                            )
                            # Match PyTorch exactly: instance_masks[candidates[idx]].squeeze(0)
                            instance_mask = instance_masks[actual_idx]

                            # Robust squeeze to handle any extra dimensions - match PyTorch .squeeze(0)
                            while (
                                instance_mask.ndim > 2 and instance_mask.shape[0] == 1
                            ):
                                instance_mask = instance_mask.squeeze(0)

                            pred_mask_bin = (
                                instance_mask > self.mask_threshold
                            ).astype("float32")

                            # Get box coordinates first, before checking if mask is valid
                            xtl, ytl, xbr, ybr = instance_boxes[actual_idx]

                            if pred_mask_bin.sum() > 0:
                                # Simple assignment like PyTorch - should work now with correct shapes
                                new_mask[ytl:ybr, xtl:xbr] = pred_mask_bin

                                # Check for overlap - match PyTorch logic exactly
                                if ((new_mask != 0) * (panoptic_mask != 0)).any():
                                    n_total = (new_mask != 0).sum()
                                    non_overlaping_mask = (new_mask != 0) * (
                                        panoptic_mask == 0
                                    )
                                    n_new = non_overlaping_mask.sum().astype("float32")
                                    if n_new / n_total > self.min_remain:
                                        # Direct assignment like PyTorch - fix data flow issue
                                        panoptic_mask = paddle.where(
                                            non_overlaping_mask,
                                            paddle.full_like(panoptic_mask, n + 1),
                                            panoptic_mask,
                                        )

                                        # Semantic assignment - match PyTorch exactly using advanced indexing
                                        sem_values = sem[actual_idx]  # [num_classes]
                                        # PyTorch: semantic_mask[:, non_overlaping_mask] = sem[candidates[idx]][:, None]
                                        # Find positions where mask is True
                                        mask_positions = paddle.nonzero(
                                            non_overlaping_mask
                                        )  # [N, 2]
                                        if len(mask_positions) > 0:
                                            # Extract coordinates
                                            h_coords = mask_positions[:, 0]  # [N]
                                            w_coords = mask_positions[:, 1]  # [N]
                                            # Assign semantic values to all mask positions
                                            for i in range(self.num_classes):
                                                semantic_mask[
                                                    i, h_coords, w_coords
                                                ] = sem_values[i]
                                else:
                                    # No overlap case - direct assignment
                                    new_mask_bool = new_mask != 0
                                    panoptic_mask = paddle.where(
                                        new_mask_bool,
                                        paddle.full_like(panoptic_mask, n + 1),
                                        panoptic_mask,
                                    )

                                    # Semantic assignment - match PyTorch exactly using advanced indexing
                                    sem_values = sem[actual_idx]  # [num_classes]
                                    # PyTorch: semantic_mask[:, (new_mask != 0)] = sem[candidates[idx]][:, None]
                                    # Find positions where mask is True
                                    mask_positions = paddle.nonzero(
                                        new_mask_bool
                                    )  # [N, 2]
                                    if len(mask_positions) > 0:
                                        # Extract coordinates
                                        h_coords = mask_positions[:, 0]  # [N]
                                        w_coords = mask_positions[:, 1]  # [N]
                                        # Assign semantic values to all mask positions
                                        for i in range(self.num_classes):
                                            semantic_mask[
                                                i, h_coords, w_coords
                                            ] = sem_values[i]

                panoptic_instance.append(panoptic_mask)
                panoptic_semantic.append(semantic_mask)
            panoptic_instance = paddle.stack(panoptic_instance, axis=0)
            panoptic_semantic = paddle.stack(panoptic_semantic, axis=0)
        else:
            panoptic_instance = None
            panoptic_semantic = None

        predictions = dict(
            center_mask=center_mask,
            saliency=saliency,
            heatmap=heatmap,
            semantic=sem,
            size=size,
            confidence=confidence,
            centerness=centerness,
            instance_masks=instance_masks,
            instance_boxes=instance_boxes,
            pano_instance=panoptic_instance,
            pano_semantic=panoptic_semantic,
        )

        return predictions


class CenterExtractor(nn.Layer):
    def __init__(self):
        """
        Module for local maxima extraction
        """
        super(CenterExtractor, self).__init__()
        self.pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

    def forward(self, input, zones=None):
        """
        Args:
            input (tensor): Centerness heatmap
            zones (tensor, Optional): Tensor that defines the mapping between each pixel position and
            the "closest" center during training (see paper paragraph Centerpoint detection).
            If provided, the highest local maxima in each zone is kept. As a result at most one
            prediction is made per ground truth object.
            If not provided, all local maxima are returned.
        """
        if zones is not None:
            # Note: torch_scatter functionality needs to be implemented using native Paddle operations
            # This is a simplified implementation - may need refinement for exact equivalence
            masks = []
            for b in range(input.shape[0]):
                x = input[b].flatten()
                zones_flat = zones[b].flatten().astype("int64")

                # Group by zone indices and find max in each zone
                # This is a simplified approach - actual scatter_max would be more efficient
                unique_zones = paddle.unique(zones_flat)
                mask = paddle.zeros_like(x)

                for zone in unique_zones:
                    if zone >= 0:  # Skip invalid zones
                        zone_mask = zones_flat == zone
                        zone_values = x[zone_mask]
                        if len(zone_values) > 0:
                            max_idx_in_zone = paddle.argmax(zone_values)
                            global_indices = paddle.nonzero(zone_mask).squeeze()
                            if global_indices.ndim == 0:
                                global_indices = global_indices.unsqueeze(0)
                            max_global_idx = global_indices[max_idx_in_zone]
                            mask[max_global_idx] = 1

                # Ensure zones[b] is 2D - remove last dimension if it's 1
                zone_shape = zones[b].shape
                # print(f"Debug: original zone_shape: {zone_shape}")
                if len(zone_shape) == 3 and zone_shape[-1] == 1:
                    zone_shape = zone_shape[:-1]  # (H, W, 1) -> (H, W)
                    # print(f"Debug: adjusted zone_shape: {zone_shape}")
                reshaped_mask = mask.reshape(zone_shape)
                # print(f"Debug: reshaped_mask shape: {reshaped_mask.shape}")
                final_mask = reshaped_mask.unsqueeze(0)
                # print(f"Debug: final_mask shape after unsqueeze: {final_mask.shape}")
                masks.append(final_mask)
            centermask = paddle.stack(masks, axis=0).astype("bool")
            # print(f"Debug: centermask shape after stack: {centermask.shape}")
            # Ensure centermask is (B, H, W) - remove any singleton dimensions except batch
            while len(centermask.shape) > 3:
                if centermask.shape[1] == 1:
                    centermask = centermask.squeeze(1)
                    # print(f"Debug: centermask shape after squeeze(1): {centermask.shape}")
                else:
                    break
        else:
            centermask = input == self.pool(input)
            no_valley = input > input.mean()
            centermask = centermask * no_valley
            # Ensure centermask is (B, H, W) by squeezing channel dimension if it's 1
            if centermask.shape[1] == 1:
                centermask = centermask.squeeze(1)

        n_centers = int(centermask.sum().detach().cpu().item())
        return centermask, n_centers

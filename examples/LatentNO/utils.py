from typing import Dict
from typing import Optional

import paddle

from ppsci.metric import base


class RelLpLoss(base.Metric):
    def __init__(
        self,
        p: int,
        key: str = "y2",
        normalizer: Optional[object] = None,
        eps: float = 1e-12,
        keep_batch: bool = False,
    ):
        if keep_batch:
            raise ValueError(f"keep_batch should be False, but got {keep_batch}.")
        super(RelLpLoss, self).__init__(keep_batch)
        self.p = p
        self.key = key
        self.normalizer = normalizer
        self.eps = eps

    def forward(
        self,
        output_dict: Dict[str, paddle.Tensor],
        label_dict: Dict[str, paddle.Tensor],
        weight_dicts: Optional[Dict] = None,
    ) -> Dict[str, "paddle.Tensor"]:
        losses: Dict[str, paddle.Tensor] = {}
        for label_key in label_dict:
            pred_key = self.key if self.key in output_dict else label_key
            pred = output_dict[pred_key]
            target = label_dict[label_key]

            if self.normalizer is not None:
                pred = self.normalizer.apply_y2(pred, device="cpu", inverse=True)
                target = self.normalizer.apply_y2(target, device="cpu", inverse=True)

            error = paddle.sum(
                paddle.abs(pred - target) ** self.p,
                axis=tuple(range(1, len(pred.shape))),
            ) ** (1.0 / self.p)
            target_norm = paddle.sum(
                paddle.abs(target) ** self.p, axis=tuple(range(1, len(target.shape)))
            ) ** (1.0 / self.p)

            denom = target_norm.clip(min=self.eps)
            rloss = paddle.mean(error / denom)
            losses[label_key] = rloss

        return losses


class RelLpLoss_time(base.Metric):
    def __init__(
        self,
        p: int,
        key: str = "y2",
        normalizer: Optional[object] = None,
        eps: float = 1e-12,
        keep_batch: bool = False,
        use_full_sequence: bool = True,
    ):
        if keep_batch:
            raise ValueError(f"keep_batch should be False, but got {keep_batch}.")
        super(RelLpLoss_time, self).__init__(keep_batch)
        self.p = p
        self.key = key
        self.normalizer = normalizer
        self.eps = eps
        self.use_full_sequence = use_full_sequence  # True: use full sequence loss; False: accumulate step-wise losses

    def forward(
        self,
        output_dict: Dict[str, paddle.Tensor],
        label_dict: Dict[str, paddle.Tensor],
        weight_dicts: Optional[Dict] = None,
    ) -> Dict[str, "paddle.Tensor"]:
        losses: Dict[str, paddle.Tensor] = {}
        for label_key in label_dict:
            if f"{self.key}_steps" in output_dict and not self.use_full_sequence:
                # Method 1: Accumulate losses at each timestep (matches backpropagation loss)
                pred_stack = output_dict[f"{self.key}_steps"]
                target_full = label_dict[label_key]
                step = pred_stack.shape[2]
                num_steps = pred_stack.shape[3]

                total_loss = paddle.to_tensor(0.0)
                for s in range(num_steps):
                    pred_s = pred_stack[..., s]
                    t_start = s * step
                    t_end = t_start + step
                    tgt_s = target_full[..., t_start:t_end]

                    if self.normalizer is not None:
                        pred_s = self.normalizer.apply_y2(pred_s, inverse=True)
                        tgt_s = self.normalizer.apply_y2(tgt_s, inverse=True)

                    # Compute Lp error for current timestep
                    error = paddle.sum(
                        paddle.abs(pred_s - tgt_s) ** self.p,
                        tuple(range(1, len(pred_s.shape))),
                    ) ** (1 / self.p)
                    target_norm = paddle.sum(
                        paddle.abs(tgt_s) ** self.p, tuple(range(1, len(tgt_s.shape)))
                    ) ** (1 / self.p)
                    step_loss = paddle.mean(error / target_norm)
                    total_loss = total_loss + step_loss

                losses[label_key] = total_loss

            else:
                # Method 2: Use full sequence loss
                pred_full = (
                    output_dict[self.key]
                    if self.key in output_dict
                    else output_dict[label_key]
                )
                target_full = label_dict[label_key]

                if self.normalizer is not None:
                    pred_full = self.normalizer.apply_y2(pred_full, inverse=True)
                    target_full = self.normalizer.apply_y2(target_full, inverse=True)

                error = paddle.sum(
                    paddle.abs(pred_full - target_full) ** self.p,
                    tuple(range(1, len(pred_full.shape))),
                ) ** (1 / self.p)
                target_norm = paddle.sum(
                    paddle.abs(target_full) ** self.p,
                    tuple(range(1, len(target_full.shape))),
                ) ** (1 / self.p)
                losses[label_key] = paddle.mean(error / target_norm)

        return losses

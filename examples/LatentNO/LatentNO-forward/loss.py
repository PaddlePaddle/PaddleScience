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
        """
        p: Lp 范数的 p
        key: 用于从 output_dict / label_dict 中取出预测/目标（优先使用 self.key；若不存在则使用 label key）
        normalizer: 可选的 normalizer（应具有方法 apply_y2(tensor, device, inverse=True/False)）
                    如果提供，则在计算 loss 前对 pred 和 target 做 inverse=True（即反归一化）。
        eps: 避免除零
        """
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
            # 选择用于预测的 key：优先用 self.key（如果存在），否则用 label_key
            pred_key = self.key if self.key in output_dict else label_key

            pred = output_dict[pred_key]
            target = label_dict[label_key]

            # 如果提供了 normalizer，就在真实尺度上计算 loss（先反归一化）
            if self.normalizer is not None:
                # paddle.get_device() 返回像 "gpu:0" 或 "cpu" 的字符串，normalizer.apply_y2 期望 device 字符串
                device = paddle.get_device()
                # 注意 normalizer.apply_y2 会 reshape -> 映射 -> reshape
                pred = self.normalizer.apply_y2(pred, device, inverse=True)
                target = self.normalizer.apply_y2(target, device, inverse=True)

            # 计算 Lp 误差：在除 batch 轴外对所有轴求和，然后开 p 次方根
            # error shape reduction: sum over dims 1..end
            error = paddle.sum(
                paddle.abs(pred - target) ** self.p,
                axis=tuple(range(1, len(pred.shape))),
            ) ** (1.0 / self.p)
            target_norm = paddle.sum(
                paddle.abs(target) ** self.p, axis=tuple(range(1, len(target.shape)))
            ) ** (1.0 / self.p)

            # 防止除零
            denom = target_norm.clip(min=self.eps)
            rloss = paddle.mean(error / denom)

            losses[label_key] = rloss

        return losses

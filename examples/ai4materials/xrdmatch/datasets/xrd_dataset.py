try:
    from ppsci.arch import register
except ImportError:

    def register(cls):
        return cls


import numpy as np
from paddle.io import Dataset


@register
class BasicDatasetPaddle(Dataset):
    def __init__(
        self,
        algorithm=None,
        data_path=None,
        target_path=None,
        num_classes=2,
        transform=None,
        is_ulb=False,
        strong_transform=None,
        **kwargs
    ):
        super().__init__()
        self.algorithm = algorithm
        self.data = data_path if not isinstance(data_path, str) else np.load(data_path)
        self.target = (
            target_path if not isinstance(target_path, str) else np.load(target_path)
        )
        self.num_classes = num_classes
        self.transform = transform
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        # 与 torch 版 BasicDataset 完全一致的数据格式
        data = self.data[index]
        target = self.target[index]

        # 数据增强
        if self.is_ulb:
            # 无标签数据：返回弱增强和强增强两个版本
            x_ulb_w = self.transform(data)
            x_ulb_s = self.strong_transform(data) if self.strong_transform else x_ulb_w

            return {"idx_ulb": index, "x_ulb_w": x_ulb_w, "x_ulb_s": x_ulb_s}
        else:
            # 有标签数据
            x_lb = self.transform(data)
            y_lb = target

            return {"idx_lb": index, "x_lb": x_lb, "y_lb": y_lb}

    def __len__(self):
        return len(self.data)

from typing import Tuple

import numpy as np
from paddle.io import Dataset


# ====================== ToyDataset（T=365, N=24） ======================
class ToyTwoModalDataset(Dataset):
    """
    A toy multimodal dataset generator combining spatiotemporal (video-like) data
    and tabular (vector) data for multi-label binary classification tasks.

    This class simulates a multimodal input setting such as climate-health,
    exposome, or remote-sensing tasks, where each sample includes:
      - A 6D tensor representing spatiotemporal exposure data (video)
      - A 1D vector representing static or tabular features
      - A binary label vector (multi-label classification)

    The dataset is generated synthetically using Gaussian random fields
    with optional random seeds for reproducibility.

    Attributes
    ----------
    file_path : str
        Path to the dataset file (not used here but kept for interface compatibility).
    input_keys : Tuple[str, ...]
        Keys for input dictionaries (default: ("input",)).
    label_keys : Tuple[str, ...]
        Keys for label dictionaries (default: ("output",)).
    n : int
        Total number of samples.
    T : int
        Temporal dimension (e.g., number of years or months).
    C : int
        Number of exposure channels or variables.
    H : int
        Spatial height (latitude dimension).
    W : int
        Spatial width (longitude dimension).
    N : int
        Inner temporal granularity (e.g., 24 hours).
    video : np.ndarray
        Simulated spatiotemporal tensor of shape (n, T, C, H, W, N).
    vec : np.ndarray
        Simulated tabular features of shape (n, 424).
    y : np.ndarray
        Multi-label binary targets of shape (n, 4).

    Methods
    -------
    __getitem__(idx)
        Returns a tuple of dictionaries: (input_dict, label_dict, extra_dict).
    __len__()
        Returns the total number of samples in the dataset.
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...] = ("input",),
        label_keys: Tuple[str, ...] = ("output",),
        n: int = 3000,  # 样本总数
        seed: int = 0,  # 随机种子
        T: int = 12,  # 暴露日期年或者月
        C: int = 10,  # 暴露变量
        H: int = 10,  # 经纬度范围
        W: int = 10,  # 经纬度范围
        N: int = 24,  # 24小时
    ):
        super().__init__()
        ### 加input和label
        self.file_path = file_path
        self.input_keys = input_keys
        self.label_keys = label_keys
        rng = np.random.default_rng(seed)
        self.n = n
        self.T = T
        self.C = C
        self.H = H
        self.W = W
        self.N = N
        self.video = rng.normal(size=(n, T, C, H, W, N)).astype("float32")
        self.vec = rng.normal(size=(n, 424)).astype("float32")
        vid_hwn = self.video.mean(axis=(3, 4, 5))  # (n,T,C)
        vid_avg = vid_hwn.mean(axis=1)  # (n,C)
        Wv = rng.normal(size=(C, 4))
        Wt = rng.normal(size=(424, 4))
        logits = vid_avg @ Wv + self.vec @ Wt + rng.normal(scale=0.5, size=(n, 4))
        probs = 1.0 / (1.0 + np.exp(-logits))
        self.y = (probs > 0.5).astype("float32")

    def __getitem__(self, idx: int):
        ###返回三个字典
        return {"video": self.video[idx], "vec": self.vec[idx]}, {"y": self.y[idx]}, {}
        # return self.video[idx], self.vec[idx], self.y[idx]

    def __len__(self):
        return self.n

    ###在这个基础上加constraint，构建已有约束，Loss Focal BCE，valid_dataloader_cfg,加input和label

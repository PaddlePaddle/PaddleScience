# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import paddle
from datasets import load_dataset
from numpy.random import default_rng


class DGMRDataset(paddle.io.Dataset):
    """
    Dataset class for DGMR (Deep Generative Model for Radar) model.
    This open-sourced UK dataset has been mirrored to HuggingFace Datasets https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km.
    If the reader cannot load the dataset from Hugging Face, please manually download it and modify the dataset_path to the local path for loading.

    Args:
        split (str, optional): The split of the dataset, "validation" or "train". Defaults to "validation".
        NUM_INPUT_FRAMES (int, optional): Number of input frames. Defaults to 4.
        NUM_TARGET_FRAMES (int, optional): Number of target frames. Defaults to 18.
        dataset_path (str, optional): Path to the dataset. Defaults to "openclimatefix/nimrod-uk-1km".
        number (int, optional): Number of samples in the dataset. Defaults to 1000.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.DGMRDataset()
    """

    def __init__(
        self,
        split: str = "validation",
        NUM_INPUT_FRAMES: int = 4,
        NUM_TARGET_FRAMES: int = 18,
        dataset_path: str = "openclimatefix/nimrod-uk-1km",
        number: int = 1000,
    ):
        super().__init__()
        paddle.seed(42)
        self.NUM_INPUT_FRAMES = NUM_INPUT_FRAMES
        self.NUM_TARGET_FRAMES = NUM_TARGET_FRAMES
        self.number = number
        self.reader = load_dataset(
            dataset_path, "sample", split=split, streaming=True, trust_remote_code=True
        )
        self.iter_reader = self.reader

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = default_rng(42)
            self.iter_reader = iter(
                self.reader.shuffle(
                    seed=rng.integers(low=0, high=100000), buffer_size=1000
                )
            )
            row = next(self.iter_reader)
        radar_frames = row["radar_frames"]
        input_frames = radar_frames[
            -self.NUM_TARGET_FRAMES - self.NUM_INPUT_FRAMES : -self.NUM_TARGET_FRAMES
        ]
        target_frames = radar_frames[-self.NUM_TARGET_FRAMES :]
        return np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]), np.moveaxis(
            target_frames, [0, 1, 2, 3], [0, 2, 3, 1]
        )

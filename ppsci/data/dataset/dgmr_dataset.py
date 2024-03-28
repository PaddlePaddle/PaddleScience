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

from typing import Tuple

import numpy as np
import paddle
from numpy.random import default_rng

try:
    import datasets
except ModuleNotFoundError:
    pass


class DGMRDataset(paddle.io.Dataset):
    """
    Dataset class for DGMR (Deep Generative Model for Radar) model.
    This open-sourced UK dataset has been mirrored to HuggingFace Datasets https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km.
    If the reader cannot load the dataset from Hugging Face, please manually download it and modify the dataset_path to the local path for loading.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        split (str, optional): The split of the dataset, "validation" or "train". Defaults to "validation".
        num_input_frames (int, optional): Number of input frames. Defaults to 4.
        num_target_frames (int, optional): Number of target frames. Defaults to 18.
        dataset_path (str, optional): Path to the dataset. Defaults to "openclimatefix/nimrod-uk-1km".

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.DGMRDataset(("input", ), ("output", )) # doctest: +SKIP
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        split: str = "validation",
        num_input_frames: int = 4,
        num_target_frames: int = 18,
        dataset_path: str = "openclimatefix/nimrod-uk-1km",
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.num_input_frames = num_input_frames
        self.num_target_frames = num_target_frames
        self.reader = datasets.load_dataset(
            dataset_path, "sample", split=split, streaming=True, trust_remote_code=True
        )
        self.iter_reader = self.reader

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
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
            -self.num_target_frames - self.num_input_frames : -self.num_target_frames
        ]
        target_frames = radar_frames[-self.num_target_frames :]
        input_item = {
            self.input_keys[0]: np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1])
        }
        label_item = {
            self.label_keys[0]: np.moveaxis(target_frames, [0, 1, 2, 3], [0, 2, 3, 1])
        }
        return input_item, label_item

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
from typing import Optional

import numpy as np
import paddle
from paddle import io
from paddle import vision


class NamedArrayDataset(io.Dataset):
    """Class for Named Array Dataset.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Dict[str, np.ndarray]): Label dict.
        weight (Dict[str, np.ndarray], optional): Weight dict.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s).

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> output = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.NamedArrayDataset(input, output, weight)
    """

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Dict[str, np.ndarray],
        weight: Dict[str, np.ndarray],
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = input
        self.label = label
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(label.keys())
        self.weight = weight
        self.transforms = transforms
        self._len = len(next(iter(input.values())))

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        weight_item = {key: value[idx] for key, value in self.weight.items()}

        # TODO(sensen): Transforms may be applied on label and weight.
        if self.transforms is not None:
            input_item = self.transforms(input_item)

        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len


class IterableNamedArrayDataset(io.IterableDataset):
    """IterableNamedArrayDataset for full-data loading.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Dict[str, np.ndarray]): Label dict.
        weight (Dict[str, np.ndarray]): Weight dict.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> label = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.IterableNamedArrayDataset(input, label, weight)
    """

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Dict[str, np.ndarray],
        weight: Dict[str, np.ndarray],
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = {key: paddle.to_tensor(value) for key, value in input.items()}
        self.label = {key: paddle.to_tensor(value) for key, value in label.items()}
        self.weight = {key: paddle.to_tensor(value) for key, value in weight.items()}
        self._len = len(next(iter(self.input.values())))

    @property
    def num_samples(self):
        """Number of samples within current dataset."""
        return self._len

    def __iter__(self):
        yield self.input, self.label, self.weight

    def __len__(self):
        return 1

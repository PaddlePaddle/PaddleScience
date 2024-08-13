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

from typing import Optional
from typing import Tuple

import numpy as np
from paddle import io
from paddle import vision

from ppsci.utils import reader


class VtuDataset(io.Dataset):
    """Dataset class for .csv file.

    Args:
        file_path (str): *.vtu file path.
        input_keys (Optional[Tuple[str, ...]]): Tuple of input keys. Defaults to None.
        label_keys (Optional[Tuple[str, ...]]): Tuple of label keys. Defaults to None.
        time_step (int): Time step with unit second.
        time_index (Optional[Tuple[int, ...]]): Time index tuple in increasing order.
        labels : Temporary variable for [load_vtk_with_time_file].
        transforms (vision.Compose, optional): Compose object contains sample wise.
            transform(s).
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Optional[Tuple[str, ...]] = None,
        label_keys: Optional[Tuple[str, ...]] = None,
        time_step: int = None,
        time_index: Optional[Tuple[int, ...]] = None,
        labels=None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()

        # load data from file
        if time_step is not None and time_index is not None:
            _input, _label = reader.load_vtk_file(
                file_path, time_step, time_index, input_keys, label_keys
            )
            _label = {key: _label[key] for key in label_keys}
        elif time_step is None and time_index is None:
            _input = reader.load_vtk_with_time_file(file_path)
            _label = {}
            for key, value in labels.items():
                if isinstance(value, (int, float)):
                    _label[key] = np.full_like(
                        next(iter(_input.values())), value, "float32"
                    )
                else:
                    _label[key] = value
        else:
            raise ValueError(
                "Error, read vtu with time_step and time_index, or neither"
            )

        # transform
        _input = transforms(_input)
        _label = transforms(_label)

        self.input = _input
        self.label = _label
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.transforms = transforms
        self.num_samples = len(next(iter(self.input.values())))

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        return (input_item, label_item, {})

    def __len__(self):
        return self.num_samples

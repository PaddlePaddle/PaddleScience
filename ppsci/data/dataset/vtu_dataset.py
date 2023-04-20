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

import types
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from paddle import io
from paddle import vision

from ppsci.utils import misc
from ppsci.utils import reader


class VtuDataset(io.Dataset):
    """Dataset class for .csv file.

    Args:
        file_path (str): *.vtu file path.
        input_keys (Tuple[str, ...]): List of input keys.
        label_keys (Tuple[str, ...]): List of input keys.
        alias_dict (Dict[str, str]): Dict of alias(es) for input and label keys.
        weight_dict (Dict[str, Union[Callable, float]], optional): Define the weight of
            each constraint variable. Defaults to None.
        timestamps (Tuple[float, ...], optional): The number of repetitions of the data
            in the time dimension. Defaults to None.
        transforms (vision.Compose, optional): Compose object contains sample wise
            transform(s).
    """

    def __init__(
        self,
        file_path: str,
        label_keys,
        time_step=None,
        time_index=None,
        labels=None,
        transforms: vision.Compose = None,
    ):
        super().__init__()

        # load data from file
        if labels is None:
            _input, _label = reader.load_vtk_file(file_path, time_step, time_index)
            _label = {key: _label[key] for key in label_keys}
        else:
            _input = reader.load_vtk_withtime_file(file_path)
            _label = {}
            for key, value in labels.items():
                if isinstance(value, (int, float)):
                    _label[key] = np.full_like(
                        next(iter(_input.values())), float(value), "float32"
                    )
                else:
                    _label[key] = value

        # transform
        _input = transforms(_input)
        _label = transforms(_label)

        self.input = _input
        self.label = _label
        self.input_keys = [key for key in self.input]
        self.label_keys = label_keys

        # prepare weights
        self.weight = {
            key: np.ones_like(next(iter(self.label.values()))) for key in self.label
        }
        self.transforms = transforms
        self.num_samples = self.check_input(self.input)

    def check_input(self, input):
        len_input = set()
        for _, value in input.items():
            len_input.add(len(value))
        if len(len_input) is not 1:
            raise AttributeError("Input dimension mismatch")
        else:
            return list(len_input)[0]

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        return (input_item, label_item)

    def __len__(self):
        return self.num_samples

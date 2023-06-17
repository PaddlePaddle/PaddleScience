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

import abc
from typing import Callable
from typing import Dict

import numpy as np


class Visualizer:
    """Base class for visualizer.

    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int): Batch size of data when computing result in visu.py.
        num_timestamps (int): Number of timestamps.
        prefix (str): Prefix for output file.
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int,
        num_timestamps: int,
        prefix: str,
    ):
        self.input_dict = input_dict
        self.input_keys = tuple(input_dict.keys())
        self.output_expr = output_expr
        self.output_keys = tuple(output_expr.keys())
        self.batch_size = batch_size
        self.num_timestamps = num_timestamps
        self.prefix = prefix

    @abc.abstractmethod
    def save(self, data_dict):
        """Visualize result from data_dict and save as files"""

    def __str__(self):
        return ", ".join(
            [
                f"input_keys: {self.input_keys}",
                f"output_keys: {self.output_keys}",
                f"output_expr: {self.output_expr}",
                f"batch_size: {self.batch_size}",
                f"num_timestamps: {self.num_timestamps}",
                f"output file prefix: {self.prefix}",
            ]
        )

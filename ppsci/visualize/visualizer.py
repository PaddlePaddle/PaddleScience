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

from ppsci.visualize import plot
from ppsci.visualize import vtu


class Visualizer(object):
    def __init__(self, input_dict, output_expr, num_timestamps, prefix):
        self.input_dict = input_dict
        self.input_keys = list(input_dict.keys())
        self.dim = len(self.input_keys)
        self.output_expr = output_expr
        self.output_keys = list(output_expr.keys())
        self.num_timestamps = num_timestamps
        self.prefix = prefix

    @abc.abstractmethod
    def save(self, data_dict):
        """visualize result from data_dict and save as files"""

    def __str__(self):
        return ", ".join(
            [
                f"input_keys: {self.input_keys}",
                f"dim: {self.dim}",
                f"output_keys: {self.output_keys}",
                f"output_expr: {self.output_expr}",
                f"num_timestamps: {self.num_timestamps}",
            ]
        )


class VisualizerScatter1D(Visualizer):
    def __init__(self, input_dict, output_expr, num_timestamps=1, prefix="plot"):
        super().__init__(input_dict, output_expr, num_timestamps, prefix)

    def save(self, data_dict, filename):
        plot.save_plot_from_1d_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )


class VisualizerVtu(Visualizer):
    def __init__(self, input_dict, output_expr, num_timestamps=1, prefix="vtu"):
        super().__init__(input_dict, output_expr, num_timestamps, prefix)

    def save(self, filename, data_dict):
        vtu.save_vtu_from_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )


class Visualizer3D(Visualizer):
    def __init__(self, input_dict, output_expr, num_timestamps=1, prefix="plot3d"):
        super().__init__(input_dict, output_expr, num_timestamps, prefix)

    def save(self, filename, data_dict):
        vtu.save_vtu_from_dict(
            filename, data_dict, self.input_keys, self.output_keys, self.num_timestamps
        )

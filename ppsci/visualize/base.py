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


class Visualizer(object):
    def __init__(self, filename, coord_keys, values_keys, num_timestamps):
        self.filename = filename
        self.coord_keys = coord_keys
        self.dim = len(coord_keys)
        self.values_keys = values_keys
        self.num_timestamps = num_timestamps

    @abc.abstractmethod
    def save(self, data_dict):
        """visualize result from data_dict and save as files"""

    def __str__(self):
        return ", ".join(
            [
                f"filename: {self.filename}",
                f"coord_keys: {self.coord_keys}",
                f"dim: {self.dim}",
                f"values_keys: {self.values_keys}",
                f"num_timestamps: {self.num_timestamps}",
            ]
        )

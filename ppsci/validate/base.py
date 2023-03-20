"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from ppsci import data


class Validator(object):
    """Base class for validators"""

    def __init__(self, dataset, dataloader_cfg, loss, metric, name):
        self.data_loader = data.build_dataloader(dataset, dataloader_cfg)
        self.data_iter = iter(self.data_loader)
        self.loss = loss
        self.metric = metric
        self.name = name

    def __str__(self):
        _str = ", ".join(
            [
                self.__class__.__name__,
                f"name = {self.name}",
                f"input_keys = {self.input_keys}",
                f"output_keys = {self.output_keys}",
                f"label_expr = {self.label_expr}",
                f"label_dict = {self.label_dict}",
                f"len(dataloader) = {len(self.data_loader)}",
                f"loss = {self.loss}",
                f"metric = {list(self.metric.keys())}",
            ]
        )
        return _str

    def _load_csv_file(self, file_path, keys, alias_dict):
        import pandas as pd

        raw_data_frame = pd.read_csv(file_path)

        # convert to numpy array
        data_dict = {}
        for key in keys:
            if key in alias_dict:
                data_dict[alias_dict[key]] = np.asarray(
                    raw_data_frame[key], "float32"
                ).reshape([-1, 1])
            else:
                data_dict[key] = np.asarray(raw_data_frame[key], "float32").reshape(
                    [-1, 1]
                )

        return data_dict

    def _load_mat_file(self, file_path, keys, alias_dict):
        import scipy.io as sio

        raw_data = sio.loadmat(file_path)

        # convert to numpy array
        data_dict = {}
        for key in keys:
            if key in alias_dict:
                data_dict[alias_dict[key]] = np.asarray(
                    raw_data[key], "float32"
                ).reshape([-1, 1])
            else:
                data_dict[key] = np.asarray(raw_data[key], "float32").reshape([-1, 1])

        return data_dict

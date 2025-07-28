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

from __future__ import annotations

import os
from typing import Callable
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from ppsci.visualize import base


class VisualizerRadar(base.Visualizer):
    """Visualizer for NowcastNet Radar Dataset.

    Args:
        input_dict (Dict[str, np.ndarray]): Input dict.
        output_expr (Dict[str, Callable]): Output expression.
        batch_size (int, optional): Batch size of data when computing result in visu.py. Defaults to 64.
        num_timestamps (int, optional): Number of timestamps
        prefix (str, optional): Prefix for output file.
        case_type (str, optional): Case type.
        total_length (str, optional): Total length.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> frames_tensor = paddle.randn([1, 29, 512, 512, 2])
        >>> visualizer =  ppsci.visualize.VisualizerRadar(
        ...     {"input": frames_tensor},
        ...     {"output": lambda out: out["output"]},
        ...     num_timestamps=1,
        ...     prefix="v_nowcastnet",
        ... )
    """

    def __init__(
        self,
        input_dict: Dict[str, np.ndarray],
        output_expr: Dict[str, Callable],
        batch_size: int = 64,
        num_timestamps: int = 1,
        prefix: str = "vtu",
        case_type: str = "normal",
        total_length: int = 29,
    ):
        super().__init__(input_dict, output_expr, batch_size, num_timestamps, prefix)
        self.case_type = case_type
        self.total_length = total_length
        self.input_dict = input_dict

    def save(self, path, data_dict):
        if not os.path.exists(path):
            os.makedirs(path)
        test_ims = self.input_dict[list(self.input_dict.keys())[0]]
        # keys: {"input", "output"}
        img_gen = data_dict[list(data_dict.keys())[1]]
        vis_info = {"vmin": 1, "vmax": 40}
        if self.case_type == "normal":
            test_ims_plot = test_ims[0][
                :-2, 256 - 192 : 256 + 192, 256 - 192 : 256 + 192
            ]
            img_gen_plot = img_gen[0][:-2, 256 - 192 : 256 + 192, 256 - 192 : 256 + 192]
        else:
            test_ims_plot = test_ims[0][:-2]
            img_gen_plot = img_gen[0][:-2]
        save_plots(
            test_ims_plot,
            labels=[f"gt{i + 1}" for i in range(self.total_length)],
            res_path=path,
            vmin=vis_info["vmin"],
            vmax=vis_info["vmax"],
        )
        save_plots(
            img_gen_plot,
            labels=[f"pd{i + 1}" for i in range(9, self.total_length)],
            res_path=path,
            vmin=vis_info["vmin"],
            vmax=vis_info["vmax"],
        )


def save_plots(
    field,
    labels,
    res_path,
    figsize=None,
    vmin=0,
    vmax=10,
    cmap="viridis",
    npy=False,
    **imshow_args,
):
    for i, data in enumerate(field):
        if i >= len(labels):
            break
        plt.figure(figsize=figsize)
        ax = plt.axes()
        ax.set_axis_off()
        alpha = data[..., 0] / 1
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1
        ax.imshow(
            data[..., 0], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args
        )
        plt.savefig(os.path.join(res_path, labels[i] + ".png"))
        plt.close()
        if npy:
            with open(os.path.join(res_path, labels[i] + ".npy"), "wb") as f:
                np.save(f, data[..., 0])

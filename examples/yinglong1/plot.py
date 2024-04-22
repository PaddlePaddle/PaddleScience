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
from typing import Dict
from typing import Tuple
from typing import Union

import imageio
import matplotlib
import numpy as np
import paddle
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle


class HandlerColormap(HandlerBase):
    """Class for creating colormap legend rectangles.

    Args:
        cmap (matplotlib.cm): Matplotlib colormap.
        num_stripes (int, optional): Number of contour levels (strips) in rectangle. Defaults to 8.
    """

    def __init__(self, cmap: matplotlib.cm, num_stripes: int = 8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle(
                [xdescent + i * width / self.num_stripes, ydescent],
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans,
            )
            stripes.append(s)
        return stripes


def _save_plot_weather_from_array(
    filename: str,
    pred: np.ndarray,
    target: np.ndarray,
    pred_key: str,
    target_key: str,
    xticks: Tuple[float, ...],
    xticklabels: Tuple[str, ...],
    yticks: Tuple[float, ...],
    yticklabels: Tuple[str, ...],
    vmin: float,
    vmax: float,
    colorbar_label: str = "",
    log_norm: bool = False,
):
    """Plot weather result as file from array data.

    Args:
        filename (str): Output file name.
        pred (np.ndarray): The predict data.
        target (np.ndarray): The target data.
        pred_key (str): The key of predict data.
        target_key (str): The key of target data.
        xticks (Tuple[float, ...]): The list of xtick locations.
        xticklabels (Tuple[str, ...]): The x-axis' tick labels.
        yticks (Tuple[float, ...]): The list of ytick locations.
        yticklabels (Tuple[str, ...]): The y-axis' tick labels.
        vmin (float): Minimal value that the colormap covers.
        vmax (float):  Maximal value that the colormap covers.
        colorbar_label (str, optional): The color-bar label. Defaults to "".
        log_norm (bool, optional): Whether use log norm. Defaults to False.
    """

    def plot_weather(
        ax,
        data,
        title_text,
        xticks,
        xticklabels,
        yticks,
        yticklabels,
        vmin,
        vmax,
        log_norm,
        cmap=cm.get_cmap("turbo", 1000),
    ):
        ax.title.set_text(title_text)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        if not log_norm:
            map_ = ax.imshow(
                data,
                interpolation="nearest",
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
            map_ = ax.imshow(
                data, interpolation="nearest", cmap=cmap, aspect="auto", norm=norm
            )
        plt.colorbar(mappable=map_, cax=None, ax=None, shrink=0.5, label=colorbar_label)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig = plt.figure(facecolor="w", figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    plot_weather(
        ax,
        pred,
        pred_key,
        xticks,
        xticklabels,
        yticks,
        yticklabels,
        vmin,
        vmax,
        log_norm,
    )
    bx = fig.add_subplot(1, 2, 2)
    plot_weather(
        bx,
        target,
        target_key,
        xticks,
        xticklabels,
        yticks,
        yticklabels,
        vmin,
        vmax,
        log_norm,
    )
    fig.savefig(filename, dpi=300)
    plt.close()


def save_plot_weather_from_dict(
    foldername: str,
    data_dict: Dict[str, Union[np.ndarray, paddle.Tensor]],
    visu_keys: Tuple[str, ...],
    xticks: Tuple[float, ...],
    xticklabels: Tuple[str, ...],
    yticks: Tuple[float, ...],
    yticklabels: Tuple[str, ...],
    vmin: float,
    vmax: float,
    colorbar_label: str = "",
    log_norm: bool = False,
    num_timestamps: int = 1,
):
    """Plot weather result as file from dict data.

    Args:
        foldername (str): Output folder name.
        data_dict (Dict[str, Union[np.ndarray, paddle.Tensor]]): Data in dict.
        visu_keys (Tuple[str, ...]): Keys for visualizing data. such as ("output_6h", "target_6h").
        xticks (Tuple[float, ...]): The list of xtick locations.
        xticklabels (Tuple[str, ...]): The x-axis' tick labels.
        yticks (Tuple[float, ...]): The list of ytick locations,
        yticklabels (Tuple[str, ...]): The y-axis' tick labels.
        vmin (float): Minimal value that the colormap covers.
        vmax (float): Maximal value that the colormap covers.
        colorbar_label (str, optional): The colorbar label. Defaults to "".
        log_norm (bool, optional): Whether use log norm. Defaults to False.
        num_timestamps (int): Number of timestamp in data_dict. Defaults to 1.
    """
    os.makedirs(foldername, exist_ok=True)

    visu_data = [data_dict[k] for k in visu_keys]
    if isinstance(visu_data[0], paddle.Tensor):
        visu_data = [x.numpy() for x in visu_data]

    frames = []
    for t in range(num_timestamps):
        pred_key, target_key = visu_keys[2 * t], visu_keys[2 * t + 1]
        pred_data = visu_data[2 * t]
        target_data = visu_data[2 * t + 1]
        filename_t = os.path.join(foldername, f"{t}.png")
        _save_plot_weather_from_array(
            filename_t,
            pred_data,
            target_data,
            pred_key,
            target_key,
            xticks,
            xticklabels,
            yticks,
            yticklabels,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
            log_norm=log_norm,
        )
        frames.append(imageio.imread(filename_t))
    filename = os.path.join(foldername, "result.gif")
    imageio.mimsave(
        filename,
        frames,
        "GIF",
        duration=1,
    )

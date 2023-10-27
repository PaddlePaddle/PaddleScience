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

import math
import os
import pathlib
import warnings
from os import path as osp
from typing import BinaryIO
from typing import List
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.vision import transforms as T
from PIL import Image

matplotlib.use("Agg")


@paddle.no_grad()
def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs,
) -> paddle.Tensor:
    if not (
        isinstance(tensor, paddle.Tensor)
        or (
            isinstance(tensor, list)
            and all(isinstance(t, paddle.Tensor) for t in tensor)
        )
    ):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[
                :,
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = (
        paddle.clip(grid * 255 + 0.5, 0, 255).transpose([1, 2, 0]).cast("uint8").numpy()
    )
    im = Image.fromarray(ndarr)
    os.makedirs(osp.dirname(fp), exist_ok=True)
    im.save(fp, format=format)


def log_images(
    nodes,
    pred,
    true,
    elems_list,
    index,
    mode,
    aoa=0,
    mach=0,
    file="field.png",
):
    for field in range(pred.shape[1]):
        true_img = plot_field(
            nodes,
            elems_list,
            true[:, field],
            mode=mode,
            col=field,
            clim=(-0.8, 0.8),
            title="true",
        )
        true_img = T.ToTensor()(true_img)

        pred_img = plot_field(
            nodes,
            elems_list,
            pred[:, field],
            mode=mode,
            col=field,
            clim=(-0.8, 0.8),
            title="pred",
        )
        pred_img = T.ToTensor()(pred_img)
        imgs = [pred_img, true_img]
        grid = make_grid(paddle.stack(imgs), padding=0)
        out_file = file + f"{field}"
        if mode == "airfoil":
            if aoa == 8.0 and mach == 0.65:
                save_image(
                    grid, "./result/image/" + str(index) + out_file + "_field.png"
                )
            save_image(
                grid, "./result/image/airfoil/" + str(index) + out_file + "_field.png"
            )
        elif mode == "cylinder":
            if aoa == 39.0:
                save_image(
                    grid, "./result/image/" + str(index) + out_file + "_field.png"
                )
            save_image(
                grid, "./result/image/cylinder/" + str(index) + out_file + "_field.png"
            )
        else:
            raise ValueError(
                f"Argument 'mode' should be 'airfoil' or 'cylinder', but got {mode}."
            )


def plot_field(
    nodes: paddle.Tensor,
    elems_list,
    field: paddle.Tensor,
    mode,
    col,
    contour=False,
    clim=None,
    zoom=True,
    get_array=True,
    out_file=None,
    show=False,
    title="",
):
    elems_list = sum(elems_list, [])
    tris, _ = quad2tri(elems_list)
    tris = np.array(tris)
    x, y = nodes[:, :2].t().detach().numpy()
    field = field.detach().numpy()
    fig = plt.figure(dpi=800)
    if contour:
        plt.tricontourf(x, y, tris, field)
    else:
        plt.tripcolor(x, y, tris, field)
    if clim:
        plt.clim(*clim)
    colorbar = plt.colorbar()
    if mode == "airfoil":
        if col == 0:
            colorbar.set_label("x-velocity", fontsize=16)
        elif col == 1:
            colorbar.set_label("pressure", fontsize=16)
        elif col == 2:
            colorbar.set_label("y-velocity", fontsize=16)
    if mode == "cylinder":
        if col == 0:
            colorbar.set_label("pressure", fontsize=16)
        elif col == 1:
            colorbar.set_label("x-velocity", fontsize=16)
        elif col == 2:
            colorbar.set_label("y-velocity", fontsize=16)
    if zoom:
        if mode == "airfoil":
            plt.xlim(left=-0.5, right=1.5)
            plt.ylim(bottom=-0.5, top=0.5)
        else:
            plt.xlim(left=-5, right=5.0)
            plt.ylim(bottom=-5, top=5.0)

    if title:
        plt.title(title)

    if out_file is not None:
        plt.savefig(out_file)
        plt.close()

    if show:
        plt.show()

    if get_array:
        if mode == "airfoil":
            plt.gca().invert_yaxis()
        fig.canvas.draw()
        array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        array = array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        fig.clear()
        plt.close()
        return array


def quad2tri(elems):
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e) <= 3:
            new_elems.append(e)
        else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(paddle.to_tensor(([[e[0]], [e[2]]]), dtype=paddle.int64))
    new_edges = (
        paddle.concat(new_edges, axis=1)
        if new_edges
        else paddle.to_tensor([], dtype=paddle.int64)
    )
    return new_elems, new_edges

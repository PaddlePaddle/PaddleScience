# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import h5py
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm


def draw_heatmap(pred,
                 gt,
                 save_name,
                 hours,
                 min=0,
                 max=25,
                 linear_scale=True,
                 colorbar_label=None):
    if not linear_scale:
        norm = matplotlib.colors.LogNorm(vmin=min, vmax=max, clip=True)
    #cmap=cm.Blues      
    cmap = cm.get_cmap('turbo', 1000)
    figure = plt.figure(facecolor='w', figsize=(7, 7))
    ax = figure.add_subplot(2, 1, 1)
    ax.title.set_text('FourCastNet: lead time = {} hours'.format(hours))
    ax.set_yticks([0, 120, 240, 360, 480, 600, 719],
                  [90, 60, 30, 0, -30, -60, -90])
    ax.set_xticks([i for i in range(0, 1440, 120)] + [1439],
                  [i for i in range(360, -1, -30)])
    if linear_scale:
        map = ax.imshow(
            pred,
            interpolation='nearest',
            cmap=cmap,
            aspect='auto',
            vmin=min,
            vmax=max)
    else:
        map = ax.imshow(
            pred, interpolation='nearest', cmap=cmap, aspect='auto', norm=norm)
    cb = plt.colorbar(
        mappable=map, cax=None, ax=None, shrink=0.5, label=colorbar_label)

    bx = figure.add_subplot(2, 1, 2)
    bx.title.set_text('ERA5: lead time = {} hours'.format(hours))
    bx.set_yticks([0, 120, 240, 360, 480, 600, 719],
                  [90, 60, 30, 0, -30, -60, -90])
    bx.set_xticks([i for i in range(0, 1440, 120)] + [1439],
                  [i for i in range(360, -1, -30)])
    if linear_scale:
        map = bx.imshow(
            gt,
            interpolation='nearest',
            cmap=cmap,
            aspect='auto',
            vmin=min,
            vmax=max)
    else:
        map = bx.imshow(
            gt, interpolation='nearest', cmap=cmap, aspect='auto', norm=norm)
    cb = plt.colorbar(
        mappable=map, cax=None, ax=bx, shrink=0.5, label=colorbar_label)
    plt.savefig(save_name)
    plt.close()


def visu_wind(file_path, save_path, means, stds):

    os.makedirs(save_path, exist_ok=True)

    means = np.asarray([means])
    stds = np.asarray([stds])

    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key == 'ground_truth':
                gt = np.asarray(f[key])
                gt = gt * stds + means
            elif key == 'predicted':
                pred = np.asarray(f[key])
                pred = pred * stds + means
    prediction_length = gt.shape[1]
    frames = []
    # import pdb;pdb.set_trace()
    for iter in range(prediction_length):
        pred_i = (pred[0, iter, 0]**2 + pred[0, iter, 1]**2)**0.5
        gt_i = (gt[0, iter, 0]**2 + gt[0, iter, 1]**2)**0.5
        hours = (iter + 1) * 6
        draw_heatmap(
            pred_i,
            gt_i,
            os.path.join(save_path, '{}.png'.format(iter)),
            hours,
            min=0,
            max=25,
            colorbar_label='m/s')
        frames.append(
            imageio.imread(os.path.join(save_path, '{}.png'.format(iter))))
    imageio.mimsave(
        os.path.join(save_path, 'result.gif'),
        frames,
        'GIF',
        duration=1, )


def visu_precip(file_path, save_path):

    os.makedirs(save_path, exist_ok=True)

    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key == 'ground_truth':
                gt = np.asarray(f[key])
            elif key == 'predicted':
                pred = np.asarray(f[key])
    gt, pred = gt[0], pred[0]
    prediction_length = gt.shape[0]

    frames = []
    for iter in range(prediction_length):
        pred_i = pred[iter, 0] * 1000
        gt_i = gt[iter, 0] * 1000
        hours = (iter + 1) * 6
        draw_heatmap(
            pred_i,
            gt_i,
            os.path.join(save_path, '{}.png'.format(iter)),
            hours,
            min=0.001,
            max=130,
            linear_scale=False,
            colorbar_label='mm')
        frames.append(
            imageio.imread(os.path.join(save_path, '{}.png'.format(iter))))
    imageio.mimsave(
        os.path.join(save_path, 'result.gif'),
        frames,
        'GIF',
        duration=1, )

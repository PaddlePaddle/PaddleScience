# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
# Ashesh Chattopadhyay - Rice University 
# Morteza Mardani - NVIDIA Corporation 
# Thorsten Kurth - NVIDIA Corporation 
# David Hall - NVIDIA Corporation 
# Zongyi Li - California Institute of Technology, NVIDIA Corporation 
# Kamyar Azizzadenesheli - Purdue University 
# Pedram Hassanzadeh - Rice University 
# Karthik Kashinath - NVIDIA Corporation 
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import numpy as np
import paddle


def unlog_tp(x, eps=1E-5):
    return eps * (np.exp(x) - 1)


def unlog_tp_paddle(x, eps=1E-5):
    return eps * (paddle.exp(x) - 1)


def mean(x, axis=None):
    #spatial mean
    y = np.sum(x, axis) / np.size(x, axis)
    return y


def lat_np(j, num_lat):
    return 90 - j * 180 / (num_lat - 1)


def weighted_acc(pred, target, weighted=True):
    #takes in shape [1, num_lat, num_long]
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)

    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(
        latitude_weighting_factor(np.arange(0, num_lat), num_lat, s),
        -1) if weighted else 1
    r = (weight * pred * target).sum() / np.sqrt(
        (weight * pred * pred).sum() * (weight * target * target).sum())
    return r


def weighted_acc_masked(pred, target, weighted=True, maskarray=1):
    #takes in shape [1, num_lat, num_long]
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)

    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    pred -= mean(pred)
    target -= mean(target)
    s = np.sum(np.cos(np.pi / 180 * lat(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(
        latitude_weighting_factor(np.arange(0, num_lat), num_lat, s),
        -1) if weighted else 1
    r = (maskarray * weight * pred * target).sum() / np.sqrt(
        (maskarray * weight * pred * pred).sum() *
        (maskarray * weight * target * target).sum())
    return r


def weighted_rmse(pred, target):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)
    #takes in arrays of size [1, h, w]  and returns latitude-weighted rmse
    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(
        latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1)
    return np.sqrt(1 / num_lat * 1 / num_long *
                   np.sum(np.dot(weight.T, (pred[0] - target[0])**2)))


def latitude_weighting_factor(j, num_lat, s):
    return num_lat * np.cos(np.pi / 180. * lat_np(j, num_lat)) / s


def top_quantiles_error(pred, target):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)
    qs = 100
    qlim = 5
    qcut = 0.1
    qtile = 1. - np.logspace(-qlim, -qcut, num=qs)
    P_tar = np.quantile(target, q=qtile, axis=(1, 2))
    P_pred = np.quantile(pred, q=qtile, axis=(1, 2))
    return np.mean(P_pred - P_tar, axis=0)


def lat(j: paddle.Tensor, num_lat: int) -> paddle.Tensor:
    return 90. - j * 180. / float(num_lat - 1)


def latitude_weighting_factor_paddle(j: paddle.Tensor,
                                     num_lat: int,
                                     s: paddle.Tensor) -> paddle.Tensor:
    return num_lat * paddle.cos(3.1416 / 180. * lat(j, num_lat)) / s


def weighted_rmse_paddle_channels(pred: paddle.Tensor,
                                  target: paddle.Tensor) -> paddle.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    lat_t = paddle.arange(start=0, end=num_lat)

    s = paddle.sum(paddle.cos(3.1416 / 180. * lat(lat_t, num_lat)))
    weight = paddle.reshape(
        latitude_weighting_factor_paddle(lat_t, num_lat, s), (1, 1, -1, 1))
    result = paddle.sqrt(
        paddle.mean(
            weight * (pred - target)**2., axis=(-1, -2)))
    return result


def weighted_rmse_paddle(pred: paddle.Tensor,
                         target: paddle.Tensor) -> paddle.Tensor:
    result = weighted_rmse_paddle_channels(pred, target)
    return paddle.mean(result, axis=0)


def weighted_acc_masked_paddle_channels(
        pred: paddle.Tensor, target: paddle.Tensor,
        maskarray: paddle.Tensor) -> paddle.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = paddle.arange(start=0, end=num_lat)
    s = paddle.sum(paddle.cos(3.1416 / 180. * lat(lat_t, num_lat)))
    weight = paddle.reshape(
        latitude_weighting_factor_paddle(lat_t, num_lat, s), (1, 1, -1, 1))
    result = paddle.sum(maskarray * weight * pred * target, axis=(
        -1, -2)) / paddle.sqrt(
            paddle.sum(maskarray * weight * pred * pred, axis=(-1, -2)) *
            paddle.sum(maskarray * weight * target * target, axis=(-1, -2)))
    return result


def weighted_acc_paddle_channels(pred: paddle.Tensor,
                                 target: paddle.Tensor) -> paddle.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = paddle.arange(start=0, end=num_lat)
    s = paddle.sum(paddle.cos(3.1416 / 180. * lat(lat_t, num_lat)))
    weight = paddle.reshape(
        latitude_weighting_factor_paddle(lat_t, num_lat, s), (1, 1, -1, 1))
    result = paddle.sum(weight * pred * target, axis=(-1, -2)) / paddle.sqrt(
        paddle.sum(weight * pred * pred, axis=(-1, -2)) *
        paddle.sum(weight * target * target, axis=(-1, -2)))
    return result


def weighted_acc_paddle(pred: paddle.Tensor,
                        target: paddle.Tensor) -> paddle.Tensor:
    result = weighted_acc_paddle_channels(pred, target)
    return paddle.mean(result, axis=0)


def unweighted_acc_paddle_channels(pred: paddle.Tensor,
                                   target: paddle.Tensor) -> paddle.Tensor:
    result = paddle.sum(pred * target, axis=(-1, -2)) / paddle.sqrt(
        paddle.sum(pred * pred, axis=(-1, -2)) * paddle.sum(target * target,
                                                            axis=(-1, -2)))
    return result


def unweighted_acc_paddle(pred: paddle.Tensor,
                          target: paddle.Tensor) -> paddle.Tensor:
    result = unweighted_acc_paddle_channels(pred, target)
    return paddle.mean(result, axis=0)


def top_quantiles_error_paddle(pred: paddle.Tensor,
                               target: paddle.Tensor) -> paddle.Tensor:
    qs = 100
    qlim = 3
    qcut = 0.1
    n, c, h, w = pred.shape
    qtile = 1. - paddle.logspace(-qlim, -qcut, num=qs)
    qtile = qtile.numpy().astype(np.float32).tolist()
    P_tar = paddle.quantile(target.reshape([n, c, h * w]), q=qtile, axis=-1)
    P_pred = paddle.quantile(pred.reshape([n, c, h * w]), q=qtile, axis=-1)
    return paddle.mean(P_pred - P_tar, axis=0).cast(paddle.float32)

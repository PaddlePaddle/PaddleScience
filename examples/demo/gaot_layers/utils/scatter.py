# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
Scatter操作模块 - 图神经网络基础操作（替代torch_scatter）
输入: 源张量和索引 | 输出: 聚合后张量 | 地位: 基础工具，被AGNO使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from typing import Literal
from typing import Optional

import paddle


def scatter_add(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """
    Scatter add operation.

    Parameters
    ----------
    src : paddle.Tensor
        Source tensor to scatter
    index : paddle.Tensor
        Index tensor indicating where to scatter
    dim : int
        Dimension along which to scatter
    dim_size : int, optional
        Size of output dimension

    Returns
    -------
    paddle.Tensor
        Scattered tensor
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # Create output tensor
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = paddle.zeros(out_shape, dtype=src.dtype)

    # Use paddle.scatter_nd_add for efficient scatter
    if dim == 0:
        # Expand index to match src shape
        index_expanded = index.reshape([-1] + [1] * (src.ndim - 1))
        index_expanded = index_expanded.expand(src.shape)

        # Scatter add
        for i in range(src.shape[0]):
            idx = int(index[i].item())
            out[idx] += src[i]
    else:
        raise NotImplementedError(f"scatter_add only supports dim=0, got dim={dim}")

    return out


def scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """Alias for scatter_add."""
    return scatter_add(src, index, dim, dim_size)


def scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """
    Scatter mean operation.

    Parameters
    ----------
    src : paddle.Tensor
        Source tensor
    index : paddle.Tensor
        Index tensor
    dim : int
        Dimension to scatter along
    dim_size : int, optional
        Output size

    Returns
    -------
    paddle.Tensor
        Mean of scattered values
    """
    # Sum
    sum_out = scatter_add(src, index, dim, dim_size)

    # Count
    ones = paddle.ones_like(src)
    count = scatter_add(ones, index, dim, dim_size)
    count = paddle.maximum(count, paddle.ones_like(count))

    return sum_out / count


def scatter_max(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> tuple:
    """
    Scatter max operation.

    Parameters
    ----------
    src : paddle.Tensor
        Source tensor
    index : paddle.Tensor
        Index tensor
    dim : int
        Dimension to scatter along
    dim_size : int, optional
        Output size

    Returns
    -------
    tuple
        (max_values, argmax_indices)
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    # Create output tensors
    out_shape = list(src.shape)
    out_shape[dim] = dim_size

    out = paddle.full(out_shape, float("-inf"), dtype=src.dtype)
    arg_out = paddle.zeros(out_shape, dtype="int64")

    # Compute max
    if dim == 0:
        for i in range(src.shape[0]):
            idx = int(index[i].item())
            if src[i].max() > out[idx].max():
                out[idx] = src[i]
                arg_out[idx] = i
    else:
        raise NotImplementedError(f"scatter_max only supports dim=0, got dim={dim}")

    return out, arg_out


def segment_csr(
    src: paddle.Tensor,
    indptr: paddle.Tensor,
    reduce: Literal["sum", "mean", "max", "min"] = "sum",
) -> paddle.Tensor:
    """
    Segment CSR operation for efficient graph operations.

    Performs reduction over segments defined by CSR indptr.

    Parameters
    ----------
    src : paddle.Tensor [num_items, ...]
        Source tensor
    indptr : paddle.Tensor [num_segments + 1]
        CSR index pointer
    reduce : str
        Reduction operation

    Returns
    -------
    paddle.Tensor [num_segments, ...]
        Reduced tensor
    """
    num_segments = indptr.shape[0] - 1
    out_shape = [num_segments] + list(src.shape[1:])

    if reduce == "sum":
        out = paddle.zeros(out_shape, dtype=src.dtype)
        for i in range(num_segments):
            start = int(indptr[i].item())
            end = int(indptr[i + 1].item())
            if start < end:
                out[i] = src[start:end].sum(axis=0)

    elif reduce == "mean":
        out = paddle.zeros(out_shape, dtype=src.dtype)
        for i in range(num_segments):
            start = int(indptr[i].item())
            end = int(indptr[i + 1].item())
            if start < end:
                out[i] = src[start:end].mean(axis=0)

    elif reduce == "max":
        out = paddle.full(out_shape, float("-inf"), dtype=src.dtype)
        for i in range(num_segments):
            start = int(indptr[i].item())
            end = int(indptr[i + 1].item())
            if start < end:
                out[i] = src[start:end].max(axis=0)

    elif reduce == "min":
        out = paddle.full(out_shape, float("inf"), dtype=src.dtype)
        for i in range(num_segments):
            start = int(indptr[i].item())
            end = int(indptr[i + 1].item())
            if start < end:
                out[i] = src[start:end].min(axis=0)

    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")

    return out


def segment_softmax(src: paddle.Tensor, indptr: paddle.Tensor) -> paddle.Tensor:
    """
    Segment-wise softmax operation.

    Applies softmax independently to each segment.

    Parameters
    ----------
    src : paddle.Tensor [num_items]
        Source values
    indptr : paddle.Tensor [num_segments + 1]
        CSR index pointer

    Returns
    -------
    paddle.Tensor [num_items]
        Softmax normalized values
    """
    num_segments = indptr.shape[0] - 1
    out = paddle.zeros_like(src)

    for i in range(num_segments):
        start = int(indptr[i].item())
        end = int(indptr[i + 1].item())

        if start < end:
            segment = src[start:end]
            # Numerical stability
            segment_max = segment.max()
            segment_exp = paddle.exp(segment - segment_max)
            segment_sum = segment_exp.sum()
            out[start:end] = segment_exp / (segment_sum + 1e-8)

    return out

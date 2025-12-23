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
评估指标模块 - L1+median+chunk分组的评估指标实现
输入: 预测值和真实值 | 输出: 相对L1误差 | 地位: 评估工具，被主程序使用
维护规则: 一旦本文件有变化，应当立即更新本文件的开头注释与所在目录的README.md
"""

from typing import Dict
from typing import Optional

import paddle

EPSILON = 1e-10


def compute_batch_errors(
    gtr: paddle.Tensor, prd: paddle.Tensor, metadata: Dict
) -> paddle.Tensor:
    """
    Compute per-sample relative L1 errors per variable chunk for a batch.

    This function matches the exact computation logic of the PyTorch version
    to ensure numerical consistency.

    Parameters
    ----------
    gtr : paddle.Tensor [batch_size, time, space, var]
        Ground truth tensor
    prd : paddle.Tensor [batch_size, time, space, var]
        Predicted tensor
    metadata : Dict
        Dataset metadata including:
        - 'active_variables': List of active variable indices
        - 'global_mean': List of global means for each variable
        - 'global_std': List of global stds for each variable
        - 'chunked_variables': List of chunk IDs for each variable

    Returns
    -------
    paddle.Tensor [batch_size, num_chunks]
        Relative L1 errors per sample per variable chunk

    Notes
    -----
    Computation steps (matching PyTorch exactly):
    1. Normalize data using global mean/std
    2. Compute absolute L1 errors
    3. Sum errors over time and space dimensions
    4. Group errors by variable chunks
    5. Compute relative errors per chunk
    """
    # Get active variables
    active_vars = metadata["active_variables"]

    # Get normalization statistics
    mean = paddle.to_tensor(metadata["global_mean"], dtype=gtr.dtype)[
        active_vars
    ].reshape([1, 1, 1, -1])

    std = paddle.to_tensor(metadata["global_std"], dtype=gtr.dtype)[
        active_vars
    ].reshape([1, 1, 1, -1])

    # Map chunks to continuous indices
    original_chunks = metadata["chunked_variables"]
    chunked_vars = [original_chunks[i] for i in active_vars]
    unique_chunks = sorted(set(chunked_vars))
    chunk_map = {
        old_chunk: new_chunk for new_chunk, old_chunk in enumerate(unique_chunks)
    }
    adjusted_chunks = [chunk_map[chunk] for chunk in chunked_vars]
    num_chunks = len(unique_chunks)

    chunks = paddle.to_tensor(adjusted_chunks, dtype="int64")  # Shape: [var]

    # Normalize data
    gtr_norm = (gtr - mean) / (std + EPSILON)
    prd_norm = (prd - mean) / (std + EPSILON)

    # Compute absolute L1 errors and sum over time and space
    abs_error = paddle.abs(gtr_norm - prd_norm)  # [batch_size, time, space, var]
    error_sum = paddle.sum(abs_error, axis=(1, 2))  # [batch_size, var]

    # Sum errors per variable chunk using scatter_add
    batch_size = error_sum.shape[0]
    chunks.unsqueeze(0).expand([batch_size, -1])  # [batch_size, var]

    error_per_chunk = paddle.zeros([batch_size, num_chunks], dtype=error_sum.dtype)

    # Manual scatter_add for chunks
    for b in range(batch_size):
        for v in range(len(adjusted_chunks)):
            chunk_id = adjusted_chunks[v]
            error_per_chunk[b, chunk_id] += error_sum[b, v]

    # Compute sum of absolute values of ground truth per chunk
    gtr_abs_sum = paddle.sum(paddle.abs(gtr_norm), axis=(1, 2))  # [batch_size, var]

    gtr_sum_per_chunk = paddle.zeros([batch_size, num_chunks], dtype=gtr_abs_sum.dtype)

    for b in range(batch_size):
        for v in range(len(adjusted_chunks)):
            chunk_id = adjusted_chunks[v]
            gtr_sum_per_chunk[b, chunk_id] += gtr_abs_sum[b, v]

    # Compute relative errors per chunk
    relative_error_per_chunk = error_per_chunk / (gtr_sum_per_chunk + EPSILON)

    return relative_error_per_chunk  # [batch_size, num_chunks]


def compute_final_metric(all_relative_errors: paddle.Tensor) -> float:
    """
    Compute the final metric from accumulated relative errors.

    This matches the PyTorch implementation:
    - Compute median over samples for each chunk
    - Take mean of medians across chunks

    Parameters
    ----------
    all_relative_errors : paddle.Tensor [num_samples, num_chunks]
        Accumulated relative errors from all samples

    Returns
    -------
    float
        Final relative L1 median error metric

    Notes
    -----
    Final metric = mean(median(errors, dim=samples))
    This is the key difference from simple mean/L2 error.
    """
    # Compute median over sample axis for each chunk
    median_error_per_chunk = paddle.median(all_relative_errors, axis=0)  # [num_chunks]

    # Take mean of median errors across all chunks
    final_metric = paddle.mean(median_error_per_chunk)

    return final_metric.item()


def compute_relative_l1_error(
    pred: paddle.Tensor, true: paddle.Tensor, metadata: Optional[Dict] = None
) -> float:
    """
    Simplified relative L1 error for basic evaluation.

    This is a simplified version without chunk grouping,
    useful for quick evaluation during training.

    Parameters
    ----------
    pred : paddle.Tensor
        Predicted values
    true : paddle.Tensor
        Ground truth values
    metadata : Dict, optional
        Metadata with normalization statistics

    Returns
    -------
    float
        Relative L1 error
    """
    if metadata is not None:
        # Normalize if metadata provided
        mean = paddle.to_tensor(metadata.get("mean", 0.0), dtype=pred.dtype)
        std = paddle.to_tensor(metadata.get("std", 1.0), dtype=pred.dtype)

        pred_norm = (pred - mean) / (std + EPSILON)
        true_norm = (true - mean) / (std + EPSILON)
    else:
        pred_norm = pred
        true_norm = true

    # Compute relative L1 error
    abs_error = paddle.abs(pred_norm - true_norm)
    abs_true = paddle.abs(true_norm)

    relative_error = abs_error.sum() / (abs_true.sum() + EPSILON)

    return relative_error.item()


class MetricsAccumulator:
    """
    Accumulator for collecting errors across batches.

    Usage
    -----
    >>> accumulator = MetricsAccumulator()
    >>> for batch in dataloader:
    ...     pred, true = model(batch), batch['label']
    ...     errors = compute_batch_errors(true, pred, metadata)
    ...     accumulator.update(errors)
    >>> final_metric = accumulator.compute()
    """

    def __init__(self):
        self.all_errors = []

    def update(self, batch_errors: paddle.Tensor):
        """
        Add batch errors to accumulator.

        Parameters
        ----------
        batch_errors : paddle.Tensor [batch_size, num_chunks]
            Errors for current batch
        """
        self.all_errors.append(batch_errors)

    def compute(self) -> float:
        """
        Compute final metric from all accumulated errors.

        Returns
        -------
        float
            Final relative L1 median error
        """
        if len(self.all_errors) == 0:
            return 0.0

        # Concatenate all batch errors
        all_errors = paddle.concat(
            self.all_errors, axis=0
        )  # [total_samples, num_chunks]

        # Compute final metric
        return compute_final_metric(all_errors)

    def reset(self):
        """Reset accumulator."""
        self.all_errors = []

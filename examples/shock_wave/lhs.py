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

import numpy as np
import paddle

dtype = paddle.get_default_dtype()


def _partition(n_sample: int, bounds: np.ndarray) -> np.ndarray:
    """为各变量的变量区间按样本数量进行划分，返回划分后的各变量区间矩阵

    Args:
        n_sample (int): Number of samples.
        bounds (np.ndarray): Lower and upper bound of each variable with shape [m, 2].

    Returns:
        np.ndarray: Partion range array wieh shape [m, n, 2].
    """
    tmp = np.arange(n_sample, dtype=dtype)  # [0,1,...,n-1].
    coefficient_lower = np.stack(
        [1 - tmp / n_sample, tmp / n_sample],
        axis=1,
    )  # [n, 2]
    coefficient_upper = np.stack(
        [1 - (tmp + 1) / n_sample, (tmp + 1) / n_sample],
        axis=1,
    )  # [n, 2]
    partition_lower = coefficient_lower @ bounds.T
    partition_upper = coefficient_upper @ bounds.T

    partition_range = np.dstack((partition_lower.T, partition_upper.T))
    return partition_range


def _representative(partition_range: np.ndarray) -> np.ndarray:
    """Compute single representitive factor.

    Args:
        partition_range (np.ndarray): Partion range array wieh shape [m, n, 2].

    Returns:
        np.ndarray: Matrix of random representitive factor with shape [n, m].
    """
    nvar = partition_range.shape[0]
    nsample = partition_range.shape[1]

    coefficient_random = np.zeros((nvar, nsample, 2), dtype)
    coefficient_random[:, :, 1] = np.random.random((nvar, nsample))
    coefficient_random[:, :, 0] = 1 - coefficient_random[:, :, 1]

    inv_map_arr = partition_range * coefficient_random

    representative_random = inv_map_arr.sum(axis=2).T
    return representative_random


def _shuffle(array: np.ndarray) -> np.ndarray:
    """Shuffle samples for each variable.

    Args:
        array (np.ndarray): Array to be shuffled wit shape [n, m].

    Returns:
        np.ndarray: Shuffled array.
    """
    for i in range(array.shape[1]):
        np.random.shuffle(array[:, i])
    return array


def _parameter_array(n_sample: int, bounds: np.ndarray) -> np.ndarray:
    """Compute parameters matrix for given number of samples.

    Args:
        n_sample (int): Number of samples.
        bounds (np.ndarray): Lower and upper bound of each variable with shape [m, 2].

    Returns:
        np.ndarray: Parameters matrix.
    """
    arr = _partition(n_sample, bounds)  # [m, n, 2]
    parameters_matrix = _shuffle(_representative(arr))
    return parameters_matrix


class LHS:
    """Latin hypercube sampling.

    Args:
        n_sample (int): Number of samples.
        bounds (np.ndarray): Lower and upper bounds of each variable with shape [m, 2].
    """

    def __init__(self, n_sample: int, bounds: np.ndarray):
        self.nsample = n_sample
        self.bounds = bounds
        self.parameter_array = _parameter_array(n_sample, bounds)

    def get_sample(self):
        return self.parameter_array

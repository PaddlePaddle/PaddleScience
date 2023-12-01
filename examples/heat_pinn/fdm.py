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

import itertools

import numpy as np


def solve(n: int, l: float) -> np.ndarray:
    """
    Solves the heat equation using the finite difference method.
    Reference: https://github.com/314arhaam/heat-pinn/blob/main/codes/heatman.ipynb

    Args:
        n (int): The number of grid points in each direction.
        l (float): The length of the square domain.

    Returns:
        np.ndarray: A 2D array containing the temperature values at each grid point.
    """
    bc = {"x=-l": 75.0, "x=+l": 0.0, "y=-l": 50.0, "y=+l": 0.0}
    B = np.zeros([n, n])
    T = np.zeros([n**2, n**2])
    for k, (i, j) in enumerate(itertools.product(range(n), range(n))):
        M = np.zeros([n, n])
        M[i, j] = -4
        if i != 0:
            M[i - 1, j] = 1
        else:
            B[i, j] += -bc["y=-l"]
        if i != n - 1:
            M[i + 1, j] = 1
        else:
            B[i, j] += -bc["y=+l"]
        if j != 0:
            M[i, j - 1] = 1
        else:
            B[i, j] += -bc["x=-l"]
        if j != n - 1:
            M[i, j + 1] = 1
        else:
            B[i, j] += -bc["x=+l"]
        m = np.reshape(M, (1, n**2))
        T[k, :] = m
    b = np.reshape(B, (n**2, 1))
    T = np.matmul(np.linalg.inv(T), b)
    T = T.reshape([n, n])
    return T

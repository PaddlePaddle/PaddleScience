# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
Code below is heavily based on https://github.com/lululxvi/deepxde
"""

import numpy as np

from .. import config
from .geometry import Geometry
from .sampler import sample


class Interval(Geometry):
    """Class for interval

    Args:
        l (float): Left position of interval.
        r (float): Right position of interval.
    """

    def __init__(self, l, r):
        super().__init__(1, (np.array([[l]]), np.array([[r]])), r - l)
        self.l = l
        self.r = r

    def is_inside(self, x: np.ndarray):
        return ((self.l <= x) & (x <= self.r)).flatten()

    def on_boundary(self, x: np.ndarray):
        return (np.isclose(x, self.l) | np.isclose(x, self.r)).flatten()

    def boundary_normal(self, x: np.ndarray):
        return -np.isclose(x, self.l).astype(config._dtype) + \
            np.isclose(x, self.r).astype(config._dtype)

    def uniform_points(self, n: int, boundary: bool=True):
        if boundary:
            return np.linspace(
                self.l, self.r, n, dtype=config._dtype).reshape([-1, 1])
        return np.linspace(
            self.l, self.r, n + 1, endpoint=False,
            dtype=config._dtype)[1:].reshape([-1, 1])

    def random_points(self, n: int, random: str="pseudo"):
        x = sample(n, 1, random)
        return (self.l + x * self.diam).astype(config._dtype)

    def uniform_boundary_points(self, n: int):
        if n == 1:
            return np.array([[self.l]]).astype(config._dtype)
        xl = np.full([n // 2, 1], self.l).astype(config._dtype)
        xr = np.full([n - n // 2, 1], self.r).astype(config._dtype)
        return np.concatenate((xl, xr), axis=0)

    def random_boundary_points(self, n: int, random: str="pseudo"):
        if n == 2:
            return np.array([[self.l], [self.r]]).astype(config._dtype)
        return np.random.choice([self.l, self.r], n).\
            reshape([-1, 1]).astype(config._dtype)

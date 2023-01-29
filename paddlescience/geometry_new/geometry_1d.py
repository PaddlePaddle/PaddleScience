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

from typing import Dict

import numpy as np

from .. import config
from .geometry import Geometry
from .sampler import sample


class Interval(Geometry):
    def __init__(self, l, r):
        super().__init__(1, (np.array([l]), np.array([r])), r - l)
        self.l = l
        self.r = r

    def inside(self, x: np.ndarray) -> np.ndarray:
        return np.logical_and(self.l <= x, x <= self.r).flatten()

    def on_boundary(self, x: np.ndarray) -> np.ndarray:
        return np.any(np.isclose(x, [self.l, self.r]), axis=-1)

    def distance2boundary(self, x: np.ndarray, dirn: np.ndarray):
        return x - self.l if dirn < 0 else self.r - x

    def mindist2boundary(self, x: np.ndarray):
        return min(np.amin(x - self.l), np.amin(self.r - x))

    def boundary_normal(self, x: np.ndarray):
        return -np.isclose(x, self.l).astype(config._dtype) + np.isclose(
            x, self.r)

    def uniform_points(self, n: int, boundary: bool=True):
        if boundary:
            return np.linspace(
                self.l, self.r, num=n, dtype=config._dtype)[:, None]
        return np.linspace(
            self.l, self.r, num=n + 1, endpoint=False,
            dtype=config._dtype)[1:, None]

    def log_uniform_points(self, n: int, boundary: bool=True):
        eps = 0 if self.l > 0 else np.finfo(config._dtype).eps
        l = np.log(self.l + eps)
        r = np.log(self.r + eps)
        if boundary:
            x = np.linspace(l, r, num=n, dtype=config._dtype)[:, None]
        else:
            x = np.linspace(
                l, r, num=n + 1, endpoint=False, dtype=config._dtype)[1:, None]
        return np.exp(x) - eps

    def random_points(self, n: int, random: str="pseudo"):
        x = sample(n, 1, random)
        return (self.diam * x + self.l).astype(config._dtype)

    def uniform_boundary_points(self, n: int):
        if n == 1:
            return np.array([[self.l]]).astype(config._dtype)
        xl = np.full((n // 2, 1), self.l).astype(config._dtype)
        xr = np.full((n - n // 2, 1), self.r).astype(config._dtype)
        return np.vstack((xl, xr))

    def random_boundary_points(self, n: int, random: str="pseudo"):
        if n == 2:
            return np.array([[self.l], [self.r]]).astype(config._dtype)
        return np.random.choice([self.l, self.r],
                                n)[:, None].astype(config._dtype)

    def periodic_point(self, x: np.ndarray, component: int=0):
        tmp = np.copy(x)
        tmp[np.isclose(x, self.l)] = self.r
        tmp[np.isclose(x, self.r)] = self.l
        return tmp

    def background_points(self,
                          x: np.ndarray,
                          dirn: np.ndarray,
                          dist2npt: np.ndarray,
                          shift: np.ndarray):
        """
        Args:
            dirn: -1 (left), or 1 (right), or 0 (both direction).
            dist2npt: A function which converts distance to the number of extra
                points (not including x).
            shift: The number of shift.
        """

        def background_points_left():
            dx = x[0] - self.l
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] - np.arange(
                -shift, n - shift + 1, dtype=config._dtype) * h
            return pts[:, None]

        def background_points_right():
            dx = self.r - x[0]
            n = max(dist2npt(dx), 1)
            h = dx / n
            pts = x[0] + np.arange(
                -shift, n - shift + 1, dtype=config._dtype) * h
            return pts[:, None]

        return (background_points_left()
                if dirn < 0 else background_points_right()
                if dirn > 0 else np.vstack(
                    (background_points_left(), background_points_right())))

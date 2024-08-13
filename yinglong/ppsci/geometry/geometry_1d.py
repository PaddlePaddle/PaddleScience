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

"""
Code below is heavily based on [https://github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde)
"""

import numpy as np
import paddle

from ppsci.geometry import geometry
from ppsci.geometry.sampler import sample
from ppsci.utils import misc


class Interval(geometry.Geometry):
    """Class for interval.

    Args:
        l (float): Left position of interval.
        r (float): Right position of interval.

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Interval(-1, 1)
    """

    def __init__(self, l: float, r: float):
        super().__init__(1, (np.array([[l]]), np.array([[r]])), r - l)
        self.l = l
        self.r = r

    def is_inside(self, x: np.ndarray):
        return ((self.l <= x) & (x <= self.r)).flatten()

    def on_boundary(self, x: np.ndarray):
        return (np.isclose(x, self.l) | np.isclose(x, self.r)).flatten()

    def boundary_normal(self, x: np.ndarray):
        return -np.isclose(x, self.l).astype(paddle.get_default_dtype()) + np.isclose(
            x, self.r
        ).astype(paddle.get_default_dtype())

    def uniform_points(self, n: int, boundary: bool = True):
        if boundary:
            return np.linspace(
                self.l, self.r, n, dtype=paddle.get_default_dtype()
            ).reshape([-1, 1])
        return np.linspace(
            self.l, self.r, n + 1, endpoint=False, dtype=paddle.get_default_dtype()
        )[1:].reshape([-1, 1])

    def random_points(self, n: int, random: str = "pseudo"):
        x = sample(n, 1, random)
        return (self.l + x * self.diam).astype(paddle.get_default_dtype())

    def uniform_boundary_points(self, n: int):
        if n == 1:
            return np.array([[self.l]], dtype=paddle.get_default_dtype())
        xl = np.full([n // 2, 1], self.l, dtype=paddle.get_default_dtype())
        xr = np.full([n - n // 2, 1], self.r, dtype=paddle.get_default_dtype())
        return np.concatenate((xl, xr), axis=0)

    def random_boundary_points(self, n: int, random: str = "pseudo"):
        if n == 2:
            return np.array([[self.l], [self.r]], dtype=paddle.get_default_dtype())
        return (
            np.random.choice([self.l, self.r], n)
            .reshape([-1, 1])
            .astype(paddle.get_default_dtype())
        )

    def periodic_point(self, x: np.ndarray, component: int = 0):
        x_array = misc.convert_to_array(x, self.dim_keys)
        periodic_x = x_array
        periodic_x[np.isclose(x_array, self.l)] = self.r
        periodic_x[np.isclose(x_array, self.r)] = self.l
        periodic_x_normal = self.boundary_normal(periodic_x)

        periodic_x = misc.convert_to_dict(periodic_x, self.dim_keys)
        periodic_x_normal = misc.convert_to_dict(
            periodic_x_normal, [f"normal_{k}" for k in self.dim_keys]
        )
        return {**periodic_x, **periodic_x_normal}

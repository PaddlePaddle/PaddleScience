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

import itertools
from typing import Tuple

import numpy as np
import paddle

from ppsci.geometry import geometry_2d
from ppsci.geometry import geometry_nd


class Cuboid(geometry_nd.Hypercube):
    """Class for Cuboid

    Args:
        xmin (Tuple[float, float, float]): Bottom left corner point [x0, y0, z0].
        xmax (Tuple[float, float, float]): Top right corner point [x1, y1, z1].

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
    """

    def __init__(
        self, xmin: Tuple[float, float, float], xmax: Tuple[float, float, float]
    ):
        super().__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))

    def random_boundary_points(self, n, random="pseudo"):
        pts = []
        density = n / self.area
        rect = geometry_2d.Rectangle(self.xmin[:-1], self.xmax[:-1])
        for z in [self.xmin[-1], self.xmax[-1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(
                np.hstack(
                    (u, np.full((len(u), 1), z, dtype=paddle.get_default_dtype()))
                )
            )
        rect = geometry_2d.Rectangle(self.xmin[::2], self.xmax[::2])
        for y in [self.xmin[1], self.xmax[1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(
                np.hstack(
                    (
                        u[:, 0:1],
                        np.full((len(u), 1), y, dtype=paddle.get_default_dtype()),
                        u[:, 1:],
                    )
                )
            )
        rect = geometry_2d.Rectangle(self.xmin[1:], self.xmax[1:])
        for x in [self.xmin[0], self.xmax[0]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(
                np.hstack(
                    (np.full((len(u), 1), x, dtype=paddle.get_default_dtype()), u)
                )
            )
        pts = np.vstack(pts)
        if len(pts) > n:
            return pts[np.random.choice(len(pts), size=n, replace=False)]
        return pts

    def uniform_boundary_points(self, n):
        h = (self.area / n) ** 0.5
        nx, ny, nz = np.ceil((self.xmax - self.xmin) / h).astype(int) + 1
        x = np.linspace(
            self.xmin[0], self.xmax[0], num=nx, dtype=paddle.get_default_dtype()
        )
        y = np.linspace(
            self.xmin[1], self.xmax[1], num=ny, dtype=paddle.get_default_dtype()
        )
        z = np.linspace(
            self.xmin[2], self.xmax[2], num=nz, dtype=paddle.get_default_dtype()
        )

        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = list(itertools.product(x, y))
            pts.append(
                np.hstack(
                    (u, np.full((len(u), 1), v, dtype=paddle.get_default_dtype()))
                )
            )
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = np.array(
                    list(itertools.product(x, z[1:-1])),
                    dtype=paddle.get_default_dtype(),
                )
                pts.append(
                    np.hstack(
                        (
                            u[:, 0:1],
                            np.full((len(u), 1), v, dtype=paddle.get_default_dtype()),
                            u[:, 1:],
                        )
                    )
                )
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = list(itertools.product(y[1:-1], z[1:-1]))
                pts.append(
                    np.hstack(
                        (np.full((len(u), 1), v, dtype=paddle.get_default_dtype()), u)
                    )
                )
        pts = np.vstack(pts)
        if len(pts) > n:
            return pts[np.random.choice(len(pts), size=n, replace=False)]
        return pts


class Sphere(geometry_nd.Hypersphere):
    """Class for Sphere

    Args:
        center (Tuple[float, float, float]): Center of the sphere [x0, y0, z0].
        radius (float): Radius of the sphere.
    """

    def __init__(self, center, radius):
        super().__init__(center, radius)

    def uniform_boundary_points(self, n: int):
        nl = np.arange(1, n + 1).astype(paddle.get_default_dtype())
        g = (np.sqrt(5) - 1) / 2
        z = (2 * nl - 1) / n - 1
        x = np.sqrt(1 - z**2) * np.cos(2 * np.pi * nl * g)
        y = np.sqrt(1 - z**2) * np.sin(2 * np.pi * nl * g)
        return np.stack((x, y, z), axis=-1)

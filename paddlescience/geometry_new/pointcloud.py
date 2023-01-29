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

import numpy as np

from .geometry import Geometry
from .. import config
from ..data import BatchSampler


class PointCloud(Geometry):
    """A geometry represented by a point cloud, i.e., a set of points in space.

    Args:
        points: A 2-D NumPy array. If `boundary_points` is not provided, `points` can
            include points both inside the geometry or on the boundary; if `boundary_points`
            is provided, `points` includes only points inside the geometry.
        boundary_points: A 2-D NumPy array.
        boundary_normals: A 2-D NumPy array.
    """

    def __init__(self, points, boundary_points=None, boundary_normals=None):
        self.points = np.asarray(points, dtype=config._dtype)
        self.num_points = len(points)
        self.boundary_points = None
        self.boundary_normals = None
        all_points = self.points
        if boundary_points is not None:
            self.boundary_points = np.asarray(
                boundary_points, dtype=config._dtype)
            self.num_boundary_points = len(boundary_points)
            all_points = np.vstack((self.points, self.boundary_points))
            self.boundary_sampler = BatchSampler(
                self.num_boundary_points, shuffle=True)
            if boundary_normals is not None:
                if len(boundary_normals) != len(boundary_points):
                    raise ValueError(
                        "the shape of boundary_normals should be the same as boundary_points"
                    )
                self.boundary_normals = np.asarray(
                    boundary_normals, dtype=config._dtype)
        super().__init__(
            len(points[0]),
            (np.amin(
                all_points, axis=0), np.amax(
                    all_points, axis=0)),
            np.inf, )
        self.indices = list(range(len(self.points)))
        self.ptr = 0

    def inside(self, x):
        return (np.isclose(
            (x[:, None, :] - self.points[None, :, :]), 0, atol=1e-6)
                .all(axis=2).any(axis=1))

    def on_boundary(self, x):
        if self.boundary_points is None:
            raise ValueError(
                "boundary_points must be defined to test on_boundary")
        return (np.isclose(
            (x[:, None, :] - self.boundary_points[None, :, :]),
            0,
            atol=1e-6, ).all(axis=2).any(axis=1))

    def random_points(self, n, random="pseudo") -> np.ndarray:
        if self.ptr + n >= self.num_points:
            part1 = self.num_points - self.ptr
            part2 = n - part1
            indices = self.indices[-part1:]
            np.random.shuffle(self.indices)
            indices += self.indices[:part2]
            self.ptr = part2
        else:
            indices = self.points[self.ptr:self.ptr + n]
            self.ptr += n

        return self.points[indices]

    def random_boundary_points(self, n, random="pseudo"):
        if self.boundary_points is None:
            raise ValueError(
                "boundary_points must be defined to test on_boundary")
        if n <= self.num_boundary_points:
            indices = self.boundary_sampler.get_next(n)
            return self.boundary_points[indices]

        x = np.tile(self.boundary_points, (n // self.num_boundary_points, 1))
        indices = self.boundary_sampler.get_next(n % self.num_boundary_points)
        return np.vstack((x, self.boundary_points[indices]))

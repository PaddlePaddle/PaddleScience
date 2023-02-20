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
from . import geometry


class CSGUnion(geometry.Geometry):
    """Construct an object by CSG Union."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError("{} | {} failed (dimensions do not match).".
                             format(geom1.idstr, geom2.idstr))
        super().__init__(
            geom1.dim,
            (
                np.minimum(geom1.bbox[0], geom2.bbox[0]),
                np.maximum(geom1.bbox[1], geom2.bbox[1]), ),
            geom1.diam + geom2.diam, )
        self.geom1 = geom1
        self.geom2 = geom2

    def is_inside(self, x):
        return np.logical_or(self.geom1.is_inside(x), self.geom2.is_inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(
                self.geom1.on_boundary(x), ~self.geom2.is_inside(x)),
            np.logical_and(
                self.geom2.on_boundary(x), ~self.geom1.is_inside(x)), )

    def boundary_normal(self, x):
        return np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.is_inside(x)
        )[:, np.newaxis] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom2.on_boundary(x), ~self.geom1.is_inside(
                x))[:, np.newaxis] * self.geom2.boundary_normal(x)

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:
            tmp = (np.random.rand(n, self.dim) *
                   (self.bbox[1] - self.bbox[0]) + self.bbox[0])
            tmp = tmp[self.is_inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(
                n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.is_inside(geom1_boundary_points)]

            geom2_boundary_points = self.geom2.random_boundary_points(
                n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                ~self.geom1.is_inside(geom2_boundary_points)]

            tmp = np.concatenate(
                (geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x


class CSGDifference(geometry.Geometry):
    """Construct an object by CSG Difference."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError("{} - {} failed (dimensions do not match).".
                             format(geom1.idstr, geom2.idstr))
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def is_inside(self, x):
        return np.logical_and(
            self.geom1.is_inside(x), ~self.geom2.is_inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(
                self.geom1.on_boundary(x), ~self.geom2.is_inside(x)),
            np.logical_and(
                self.geom1.is_inside(x), self.geom2.on_boundary(x)), )

    def boundary_normal(self, x):
        return np.logical_and(
            self.geom1.on_boundary(x), ~self.geom2.is_inside(x)
        )[:, np.newaxis] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom1.is_inside(x), self.geom2.on_boundary(
                x))[:, np.newaxis] * -self.geom2.boundary_normal(x)

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[~self.geom2.is_inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(
                n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.is_inside(geom1_boundary_points)]

            geom2_boundary_points = self.geom2.random_boundary_points(
                n, random=random)
            geom2_boundary_points = geom2_boundary_points[self.geom1.is_inside(
                geom2_boundary_points)]

            tmp = np.concatenate(
                (geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x


class CSGIntersection(geometry.Geometry):
    """Construct an object by CSG Intersection."""

    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError("{} & {} failed (dimensions do not match).".
                             format(geom1.idstr, geom2.idstr))
        super().__init__(
            geom1.dim,
            (
                np.maximum(geom1.bbox[0], geom2.bbox[0]),
                np.minimum(geom1.bbox[1], geom2.bbox[1]), ),
            min(geom1.diam, geom2.diam), )
        self.geom1 = geom1
        self.geom2 = geom2

    def is_inside(self, x):
        return np.logical_and(self.geom1.is_inside(x), self.geom2.is_inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(
                self.geom1.on_boundary(x), self.geom2.is_inside(x)),
            np.logical_and(
                self.geom1.is_inside(x), self.geom2.on_boundary(x)), )

    def boundary_normal(self, x):
        return np.logical_and(
            self.geom1.on_boundary(x), self.geom2.is_inside(x)
        )[:, np.newaxis] * self.geom1.boundary_normal(x) + np.logical_and(
            self.geom1.is_inside(x), self.geom2.on_boundary(
                x))[:, np.newaxis] * self.geom2.boundary_normal(x)

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.is_inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:

            geom1_boundary_points = self.geom1.random_boundary_points(
                n, random=random)
            geom1_boundary_points = geom1_boundary_points[self.geom2.is_inside(
                geom1_boundary_points)]

            geom2_boundary_points = self.geom2.random_boundary_points(
                n, random=random)
            geom2_boundary_points = geom2_boundary_points[self.geom1.is_inside(
                geom2_boundary_points)]

            tmp = np.concatenate(
                (geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

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
from __future__ import annotations

import abc
from typing import Tuple

import numpy as np
import paddle

from ppsci.utils import logger
from ppsci.utils import misc


class Geometry:
    """Base class for geometry.

    Args:
        ndim (int): Number of geometry dimension.
        bbox (Tuple[np.ndarray, np.ndarray]): Bounding box of upper and lower.
        diam (float): Diameter of geometry.
    """

    def __init__(self, ndim: int, bbox: Tuple[np.ndarray, np.ndarray], diam: float):
        self.ndim = ndim
        self.bbox = bbox
        self.diam = min(diam, np.linalg.norm(bbox[1] - bbox[0]))

    @property
    def dim_keys(self):
        return ("x", "y", "z")[: self.ndim]

    @abc.abstractmethod
    def is_inside(self, x):
        """Returns a boolean array where x is inside the geometry."""

    @abc.abstractmethod
    def on_boundary(self, x):
        """Returns a boolean array where x is on geometry boundary."""

    def boundary_normal(self, x):
        """Compute the unit normal at x."""
        raise NotImplementedError(f"{self}.boundary_normal is not implemented")

    def uniform_points(self, n: int, boundary=True):
        """Compute the equi-spaced points in the geometry."""
        logger.warning(
            f"{self}.uniform_points not implemented. " f"Use random_points instead."
        )
        return self.random_points(n)

    def sample_interior(
        self,
        n,
        random="pseudo",
        criteria=None,
        evenly=False,
        compute_sdf_derivatives: bool = False,
    ):
        """Sample random points in the geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.ndim), dtype=paddle.get_default_dtype())
        _size, _ntry, _nsuc = 0, 0, 0
        while _size < n:
            if evenly:
                points = self.uniform_points(n)
            else:
                if misc.typename(self) == "TimeXGeometry":
                    points = self.random_points(n, random, criteria)
                else:
                    points = self.random_points(n, random)

            if criteria is not None:
                criteria_mask = criteria(*np.split(points, self.ndim, axis=1)).flatten()
                points = points[criteria_mask]

            if len(points) > n - _size:
                points = points[: n - _size]
            x[_size : _size + len(points)] = points

            _size += len(points)
            _ntry += 1
            if len(points) > 0:
                _nsuc += 1

            if _ntry >= 1000 and _nsuc == 0:
                raise ValueError(
                    "Sample interior points failed, "
                    "please check correctness of geometry and given criteria."
                )

        # if sdf_func added, return x_dict and sdf_dict, else, only return the x_dict
        if hasattr(self, "sdf_func"):
            sdf = -self.sdf_func(x)
            sdf_dict = misc.convert_to_dict(sdf, ("sdf",))
            sdf_derives_dict = {}
            if compute_sdf_derivatives:
                sdf_derives = -self.sdf_derivatives(x)
                sdf_derives_dict = misc.convert_to_dict(
                    sdf_derives, tuple(f"sdf__{key}" for key in self.dim_keys)
                )
        else:
            sdf_dict = {}
            sdf_derives_dict = {}
        x_dict = misc.convert_to_dict(x, self.dim_keys)

        return {**x_dict, **sdf_dict, **sdf_derives_dict}

    def sample_boundary(self, n, random="pseudo", criteria=None, evenly=False):
        """Compute the random points in the geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.ndim), dtype=paddle.get_default_dtype())
        _size, _ntry, _nsuc = 0, 0, 0
        while _size < n:
            if evenly:
                if (
                    misc.typename(self) == "TimeXGeometry"
                    and misc.typename(self.geometry) == "Mesh"
                ):
                    points, normal, area = self.uniform_boundary_points(n)
                else:
                    points = self.uniform_boundary_points(n)
            else:
                if (
                    misc.typename(self) == "TimeXGeometry"
                    and misc.typename(self.geometry) == "Mesh"
                ):
                    points, normal, area = self.random_boundary_points(n, random)
                else:
                    if misc.typename(self) == "TimeXGeometry":
                        points = self.random_boundary_points(n, random, criteria)
                    else:
                        points = self.random_boundary_points(n, random)

            if criteria is not None:
                criteria_mask = criteria(*np.split(points, self.ndim, axis=1)).flatten()
                points = points[criteria_mask]

            if len(points) > n - _size:
                points = points[: n - _size]
            x[_size : _size + len(points)] = points

            _size += len(points)
            _ntry += 1
            if len(points) > 0:
                _nsuc += 1

            if _ntry >= 1000 and _nsuc == 0:
                raise ValueError(
                    "Sample boundary points failed, "
                    "please check correctness of geometry and given criteria."
                )

        if not (
            misc.typename(self) == "TimeXGeometry"
            and misc.typename(self.geometry) == "Mesh"
        ):
            normal = self.boundary_normal(x)

        normal_dict = misc.convert_to_dict(
            normal[:, 1:] if "t" in self.dim_keys else normal,
            [f"normal_{key}" for key in self.dim_keys if key != "t"],
        )
        x_dict = misc.convert_to_dict(x, self.dim_keys)
        if (
            misc.typename(self) == "TimeXGeometry"
            and misc.typename(self.geometry) == "Mesh"
        ):
            area_dict = misc.convert_to_dict(area[:, 1:], ["area"])
            return {**x_dict, **normal_dict, **area_dict}

        return {**x_dict, **normal_dict}

    @abc.abstractmethod
    def random_points(self, n: int, random: str = "pseudo"):
        """Compute the random points in the geometry."""

    def uniform_boundary_points(self, n: int):
        """Compute the equi-spaced points on the boundary."""
        logger.warning(
            f"{self}.uniform_boundary_points not implemented. "
            f"Use random_boundary_points instead."
        )
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random points on the boundary."""

    def periodic_point(self, x: np.ndarray, component: int):
        """Compute the periodic image of x."""
        raise NotImplementedError(f"{self}.periodic_point to be implemented")

    def sdf_derivatives(self, x: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """Compute derivatives of SDF function.

        Args:
            x (np.ndarray): Points for computing SDF derivatives using central
                difference. The shape is [N, D], D is the number of dimension of
                geometry.
            epsilon (float): Derivative step. Defaults to 1e-4.

        Returns:
            np.ndarray: Derivatives of corresponding SDF function.
                The shape is [N, D]. D is the number of dimension of geometry.
        """
        if not hasattr(self, "sdf_func"):
            raise NotImplementedError(
                f"{misc.typename(self)}.sdf_func should be implemented "
                "when using 'sdf_derivatives'."
            )
        # Only compute sdf derivatives for those already implement `sdf_func` method.
        sdf_derives = np.empty_like(x)
        for i in range(self.ndim):
            h = np.zeros_like(x)
            h[:, i] += epsilon / 2
            derives_at_i = (self.sdf_func(x + h) - self.sdf_func(x - h)) / epsilon
            sdf_derives[:, i : i + 1] = derives_at_i
        return sdf_derives

    def union(self, other):
        """CSG Union."""
        from ppsci.geometry import csg

        return csg.CSGUnion(self, other)

    def __or__(self, other):
        """CSG Union."""
        from ppsci.geometry import csg

        return csg.CSGUnion(self, other)

    def difference(self, other):
        """CSG Difference."""
        from ppsci.geometry import csg

        return csg.CSGDifference(self, other)

    def __sub__(self, other):
        """CSG Difference."""
        from ppsci.geometry import csg

        return csg.CSGDifference(self, other)

    def intersection(self, other):
        """CSG Intersection."""
        from ppsci.geometry import csg

        return csg.CSGIntersection(self, other)

    def __and__(self, other):
        """CSG Intersection."""
        from ppsci.geometry import csg

        return csg.CSGIntersection(self, other)

    def __str__(self) -> str:
        """Return the name of class"""
        return ", ".join(
            [
                self.__class__.__name__,
                f"ndim = {self.ndim}",
                f"bbox = {self.bbox}",
                f"diam = {self.diam}",
                f"dim_keys = {self.dim_keys}",
            ]
        )

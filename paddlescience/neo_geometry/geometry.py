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

import abc
from typing import Callable, Dict, Optional

import numpy as np

from .. import config
from ..utils import AttrDict


class Geometry(object):
    def __init__(self, dim: int, bbox, diam: float):
        self.dim = dim
        self.bbox = bbox
        self.diam = min(diam, np.linalg.norm(bbox[1] - bbox[0]))
        self.sample_config = {}
        self.batch_data_dict = {}

    def add_sample_config(self,
                          name: str,
                          batch_size: int,
                          criteria: Optional[Callable]=None,
                          random: str="pseudo",
                          fixed: bool=True):
        """Add configuration for sampling points in geometry.

        Args:
            name (str): Name for sampled points, such as 'interior', 'top', 'left'.
            batch_size (int): Number of points to be sampled.
            criteria (Callable, optional): Criteria sampled points meets. Defaults to None.
            random (str, optional): Method for random such as "pseudo", "LHS". Defaults to "pseudo".
            fixed (bool, optional): Whether to fix sampled points. Defaults to True.
        """
        if name == "interior" or name == "boundary":
            print(f"criteria for {name} is unneccessary, set criteria to None")
            criteria = None
        self.sample_config[name] = AttrDict({
            "criteria": criteria,
            "batch_size": batch_size,
            "random": random,
            "fixed": fixed
        })

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
        """Compute the equispaced points in the geometry."""
        print(f"Warning: {self}.uniform_points not implemented. "
              f"Use random_points instead.")
        return self.random_points(n)

    def random_points_with_criteria(self,
                                    n: int,
                                    random: str="pseudo",
                                    criteria: Callable=None) -> np.ndarray:
        """Compute the random points in the geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:
            tmp = self.random_points(n, random)
            if criteria is not None:
                criteria_mask = criteria(*np.split(
                    tmp, self.dim, axis=1)).flatten()
                tmp = tmp[criteria_mask]
            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points_with_criteria(
            self, n: int, random: str="pseudo",
            criteria: Callable=None) -> np.ndarray:
        """Compute the random points in the geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.dim), dtype=config._dtype)
        i = 0
        while i < n:
            tmp = self.random_boundary_points(n, random)
            if criteria is not None:
                criteria_mask = criteria(*np.split(
                    tmp, self.dim, axis=1)).flatten()
                tmp = tmp[criteria_mask]
            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    @abc.abstractmethod
    def random_points(self, n: int, random: str="pseudo"):
        """Compute the random points in the geometry."""

    def uniform_boundary_points(self, n: int):
        """Compute the equispaced points on the boundary."""
        print(f"Warning: {self}.uniform_boundary_points not implemented. "
              f"Use random_boundary_points instead.")
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random points on the boundary."""

    def periodic_point(self, x: np.ndarray, component: int):
        """Compute the periodic image of x."""
        raise NotImplementedError(f"{self}.periodic_point to be implemented")

    def fetch_batch_data(self) -> Dict[str, np.ndarray]:
        """Sampling points with specified sample configs"""
        batch_data_dict = {}
        # generate batch data
        for name, cfg in self.sample_config.items():
            if cfg.fixed and name in self.batch_data_dict:
                batch_data_dict[name] = self.batch_data_dict[name]
                continue
            if name == "interior":
                raw_data = self.random_points(cfg.batch_size, cfg.random)
            elif name == "boundary":
                raw_data = self.random_boundary_points(cfg.batch_size,
                                                       cfg.random)
            elif name.startswith("interior_"):
                raw_data = self.random_points_with_criteria(
                    cfg.batch_size, cfg.random, cfg.criteria)
            elif name.startswith("boundary_"):
                raw_data = self.random_boundary_points_with_criteria(
                    cfg.batch_size, cfg.random, cfg.criteria)
            else:
                raw_data = self.random_points_with_criteria(
                    cfg.batch_size, cfg.random, cfg.criteria)
            batch_data_dict[name] = raw_data

        # cache data if used later
        self.batch_data_dict = batch_data_dict

        return batch_data_dict

    def union(self, other):
        """CSG Union."""
        from . import csg

        return csg.CSGUnion(self, other)

    def __or__(self, other):
        """CSG Union."""
        from . import csg

        return csg.CSGUnion(self, other)

    def difference(self, other):
        """CSG Difference."""
        from . import csg

        return csg.CSGDifference(self, other)

    def __sub__(self, other):
        """CSG Difference."""
        from . import csg

        return csg.CSGDifference(self, other)

    def intersection(self, other):
        """CSG Intersection."""
        from . import csg

        return csg.CSGIntersection(self, other)

    def __and__(self, other):
        """CSG Intersection."""
        from . import csg

        return csg.CSGIntersection(self, other)

    def __str__(self) -> str:
        """Return the name of class"""
        return self.__class__.__name__

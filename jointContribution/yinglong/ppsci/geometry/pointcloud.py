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

from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np

from ppsci.geometry import geometry
from ppsci.utils import misc


class PointCloud(geometry.Geometry):
    """Class for point cloud geometry, i.e. a set of points from given file or array.

    Args:
        interior (Dict[str, np.ndarray]): Filepath or dict data, which store interior points of a point cloud, such as {"x": np.ndarray, "y": np.ndarray}.
        coord_keys (Tuple[str, ...]): Tuple of coordinate keys, such as ("x", "y").
        boundary (Dict[str, np.ndarray]): Boundary points of a point cloud. Defaults to None.
        boundary_normal (Dict[str, np.ndarray]): Boundary normal points of a point cloud. Defaults to None.

    Examples:
        >>> import ppsci
        >>> import numpy as np
        >>> interior_points = {"x": np.linspace(-1, 1, dtype="float32").reshape((-1, 1))}
        >>> geom = ppsci.geometry.PointCloud(interior_points, ("x",))
    """

    def __init__(
        self,
        interior: Dict[str, np.ndarray],
        coord_keys: Tuple[str, ...],
        boundary: Optional[Dict[str, np.ndarray]] = None,
        boundary_normal: Optional[Dict[str, np.ndarray]] = None,
    ):
        # Interior points
        self.interior = misc.convert_to_array(interior, coord_keys)
        self.len = self.interior.shape[0]

        # Boundary points
        self.boundary = boundary
        if self.boundary is not None:
            self.boundary = misc.convert_to_array(self.boundary, coord_keys)

        # Boundary normal points
        self.normal = boundary_normal
        if self.normal is not None:
            self.normal = misc.convert_to_array(
                self.normal, tuple(f"{key}_normal" for key in coord_keys)
            )
            if list(self.normal.shape) != list(self.boundary.shape):
                raise ValueError(
                    f"boundary's shape({self.boundary.shape}) must equal "
                    f"to normal's shape({self.normal.shape})"
                )

        self.input_keys = coord_keys
        super().__init__(
            len(coord_keys),
            (np.amin(self.interior, axis=0), np.amax(self.interior, axis=0)),
            np.inf,
        )

    @property
    def dim_keys(self):
        return self.input_keys

    def is_inside(self, x):
        # NOTE: point on boundary is included
        return (
            np.isclose((x[:, None, :] - self.interior[None, :, :]), 0, atol=1e-6)
            .all(axis=2)
            .any(axis=1)
        )

    def on_boundary(self, x):
        if not self.boundary:
            raise ValueError(
                "self.boundary must be initialized" " when call 'on_boundary' function"
            )
        return (
            np.isclose(
                (x[:, None, :] - self.boundary[None, :, :]),
                0,
                atol=1e-6,
            )
            .all(axis=2)
            .any(axis=1)
        )

    def translate(self, translation):
        for i, offset in enumerate(translation):
            self.interior[:, i] += offset
            if self.boundary:
                self.boundary += offset
        return self

    def scale(self, scale):
        for i, _scale in enumerate(scale):
            self.interior[:, i] *= _scale
            if self.boundary:
                self.boundary[:, i] *= _scale
            if self.normal:
                self.normal[:, i] *= _scale
        return self

    def uniform_boundary_points(self, n: int):
        """Compute the equispaced points on the boundary."""
        raise NotImplementedError(
            "PointCloud do not have 'uniform_boundary_points' method"
        )

    def random_boundary_points(self, n, random="pseudo"):
        assert self.boundary is not None, (
            "boundary points can't be empty when call "
            "'random_boundary_points' method"
        )
        assert n <= len(self.boundary), (
            f"number of sample points({n}) "
            f"can't be more than that in boundary({len(self.boundary)})"
        )
        return self.boundary[
            np.random.choice(len(self.boundary), size=n, replace=False)
        ]

    def random_points(self, n, random="pseudo"):
        assert n <= len(self.interior), (
            f"number of sample points({n}) "
            f"can't be more than that in points({len(self.interior)})"
        )
        return self.interior[
            np.random.choice(len(self.interior), size=n, replace=False)
        ]

    def uniform_points(self, n: int, boundary=True):
        """Compute the equispaced points in the geometry."""
        return self.interior[:n]

    def union(self, other):
        raise NotImplementedError(
            "Union operation for PointCloud is not supported yet."
        )

    def __or__(self, other):
        raise NotImplementedError(
            "Union operation for PointCloud is not supported yet."
        )

    def difference(self, other):
        raise NotImplementedError(
            "Subtraction operation for PointCloud is not supported yet."
        )

    def __sub__(self, other):
        raise NotImplementedError(
            "Subtraction operation for PointCloud is not supported yet."
        )

    def intersection(self, other):
        raise NotImplementedError(
            "Intersection operation for PointCloud is not supported yet."
        )

    def __and__(self, other):
        raise NotImplementedError(
            "Intersection operation for PointCloud is not supported yet."
        )

    def __str__(self) -> str:
        """Return the name of class"""
        return ", ".join(
            [
                self.__class__.__name__,
                f"num_points = {len(self.interior)}",
                f"ndim = {self.ndim}",
                f"bbox = {self.bbox}",
                f"dim_keys = {self.dim_keys}",
            ]
        )

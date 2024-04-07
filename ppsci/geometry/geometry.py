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
from typing import Callable
from typing import Dict
from typing import Optional
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
    def is_inside(self, x: np.ndarray) -> np.ndarray:
        """Returns a boolean array where x is inside the geometry.

        Args:
            x (np.ndarray): Points to check if inside the geometry. The shape is [N, D],
                where D is the number of dimension of geometry.

        Returns:
            np.ndarray: Boolean array where x is inside the geometry. The shape is [N].

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> x = np.array([[0], [0.5], [1.5]])
            >>> interval.is_inside(x)
            array([ True,  True, False])
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> x = np.array([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]])
            >>> rectangle.is_inside(x)
            array([ True,  True, False])
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> x = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
            >>> cuboid.is_inside(x)
            array([ True,  True, False])
        """

    @abc.abstractmethod
    def on_boundary(self, x: np.ndarray) -> np.ndarray:
        """Returns a boolean array where x is on geometry boundary.

        Args:
            x (np.ndarray): Points to check if on the geometry boundary. The shape is [N, D],
                where D is the number of dimension of geometry.

        Returns:
            np.ndarray: Boolean array where x is on the geometry boundary. The shape is [N].

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> x = np.array([[0], [0.5], [1.5]])
            >>> interval.on_boundary(x)
            array([ True, False, False])
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> x = np.array([[0, 0], [0.5, 0.5], [1, 1.5]])
            >>> rectangle.on_boundary(x)
            array([ True, False, False])
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> x = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1.5]])
            >>> cuboid.on_boundary(x)
            array([ True, False, False])
        """

    def boundary_normal(self, x):
        """Compute the unit normal at x."""
        raise NotImplementedError(f"{self}.boundary_normal is not implemented")

    def uniform_points(self, n: int, boundary: bool = True) -> np.ndarray:
        """Compute the equi-spaced points in the geometry.

        Warings:
            This function is not implemented, please use random_points instead.

        Args:
            n (int): Number of points.
            boundary (bool): Include boundary points. Defaults to True.

        Returns:
            np.ndarray: Random points in the geometry. The shape is [N, D].
        """
        logger.warning(
            f"{self}.uniform_points not implemented. " f"Use random_points instead."
        )
        return self.random_points(n)

    def sample_interior(
        self,
        n: int,
        random: str = "pseudo",
        criteria: Optional[Callable] = None,
        evenly: bool = False,
        compute_sdf_derivatives: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample random points in the geometry and return those meet criteria.

        Args:
            n (int): Number of points.
            random (str): Random method. Defaults to "pseudo".
            criteria (Optional[Callable]): Criteria function. Defaults to None.
            evenly (bool): Evenly sample points. Defaults to False.
            compute_sdf_derivatives (bool): Compute SDF derivatives. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Random points in the geometry. The shape is [N, D].
                                   their signed distance function. The shape is [N, 1].
                                   their derivatives of SDF(optional). The shape is [N, D].

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> np.random.seed(42)
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> interval.sample_interior(2)
            {'x': array([[0.37454012],
                   [0.9507143 ]], dtype=float32), 'sdf': array([[0.37454012],
                   [0.04928571]], dtype=float32)}
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> rectangle.sample_interior(2, "pseudo", None, False, True)
            {'x': array([[0.7319939 ],
                   [0.15601864]], dtype=float32), 'y': array([[0.5986585 ],
                   [0.15599452]], dtype=float32), 'sdf': array([[0.2680061 ],
                   [0.15599453]], dtype=float32), 'sdf__x': array([[-1.0001659 ],
                   [ 0.25868416]], dtype=float32), 'sdf__y': array([[-0.        ],
                   [ 0.74118376]], dtype=float32)}
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> cuboid.sample_interior(2, "pseudo", None, True, True)
            {'x': array([[0.],
                   [0.]], dtype=float32), 'y': array([[0.],
                   [0.]], dtype=float32), 'z': array([[0.],
                   [1.]], dtype=float32), 'sdf': array([[0.],
                   [0.]], dtype=float32), 'sdf__x': array([[0.50008297],
                   [0.50008297]], dtype=float32), 'sdf__y': array([[0.50008297],
                   [0.50008297]], dtype=float32), 'sdf__z': array([[ 0.50008297],
                   [-0.49948692]], dtype=float32)}
        """
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

    def sample_boundary(
        self,
        n: int,
        random: str = "pseudo",
        criteria: Optional[Callable] = None,
        evenly: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute the random points in the geometry and return those meet criteria.

        Args:
            n (int): Number of points.
            random (str): Random method. Defaults to "pseudo".
            criteria (Optional[Callable]): Criteria function. Defaults to None.
            evenly (bool): Evenly sample points. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Random points in the geometry. The shape is [N, D].
                                   their normal vectors. The shape is [N, D].
                                   their area. The shape is [N, 1].(only if the geometry is a mesh)

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> np.random.seed(42)
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> interval.sample_boundary(2)
            {'x': array([[0.],
                   [1.]], dtype=float32), 'normal_x': array([[-1.],
                   [ 1.]], dtype=float32)}
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> rectangle.sample_boundary(2)
            {'x': array([[1.],
                   [0.]], dtype=float32), 'y': array([[0.49816048],
                   [0.19714284]], dtype=float32), 'normal_x': array([[ 1.],
                   [-1.]], dtype=float32), 'normal_y': array([[0.],
                   [0.]], dtype=float32)}
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> cuboid.sample_boundary(2)
            {'x': array([[0.83244264],
                   [0.18182497]], dtype=float32), 'y': array([[0.21233912],
                   [0.1834045 ]], dtype=float32), 'z': array([[0.],
                   [1.]], dtype=float32), 'normal_x': array([[0.],
                   [0.]], dtype=float32), 'normal_y': array([[0.],
                   [0.]], dtype=float32), 'normal_z': array([[-1.],
                   [ 1.]], dtype=float32)}
        """
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
    def random_points(self, n: int, random: str = "pseudo") -> np.ndarray:
        """Compute the random points in the geometry.

        Args:
            n (int): Number of points.
            random (str): Random method. Defaults to "pseudo".

        Returns:
            np.ndarray: Random points in the geometry. The shape is [N, D].

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> np.random.seed(42)
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> interval.random_points(2)
            array([[0.37454012],
                   [0.9507143 ]], dtype=float32)
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> rectangle.random_points(2)
            array([[0.7319939 , 0.5986585 ],
                   [0.15601864, 0.15599452]], dtype=float32)
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> cuboid.random_points(2)
            array([[0.05808361, 0.8661761 , 0.601115  ],
                   [0.7080726 , 0.02058449, 0.96990985]], dtype=float32)
        """

    def uniform_boundary_points(self, n: int) -> np.ndarray:
        """Compute the equi-spaced points on the boundary(not implemented).

        Warings:
            This function is not implemented, please use random_boundary_points instead.

        Args:
            n (int): Number of points.

        Returns:
            np.ndarray: Random points on the boundary. The shape is [N, D].
        """
        logger.warning(
            f"{self}.uniform_boundary_points not implemented. "
            f"Use random_boundary_points instead."
        )
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n: int, random: str = "pseudo") -> np.ndarray:
        """Compute the random points on the boundary.

        Args:
            n (int): Number of points.
            random (str): Random method. Defaults to "pseudo".

        Returns:
            np.ndarray: Random points on the boundary. The shape is [N, D].

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> np.random.seed(42)
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> interval.random_boundary_points(2)
            array([[0.],
                   [1.]], dtype=float32)
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> rectangle.random_boundary_points(2)
            array([[1.        , 0.49816048],
                   [0.        , 0.19714284]], dtype=float32)
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> cuboid.random_boundary_points(2)
            array([[0.83244264, 0.21233912, 0.        ],
                   [0.18182497, 0.1834045 , 1.        ]], dtype=float32)
        """

    def periodic_point(self, x: np.ndarray, component: int):
        """Compute the periodic image of x(not implemented).

        Warings:
            This function is not implemented.
        """
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

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> x = np.array([[0], [0.5], [1.5]])
            >>> interval.sdf_derivatives(x)
            array([[-1.],
                   [ 0.],
                   [ 1.]])
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> x = np.array([[0.0, 0.0], [0.5, 0.5], [1.5, 1.5]])
            >>> rectangle.sdf_derivatives(x)
            array([[-0.5       , -0.5       ],
                   [ 0.        ,  0.        ],
                   [ 0.70710678,  0.70710678]])
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> x = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
            >>> cuboid.sdf_derivatives(x)
            array([[-0.5, -0.5, -0.5],
                   [ 0. ,  0. ,  0. ],
                   [ 0.5,  0.5,  0.5]])
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

    def union(self, other: "Geometry") -> "Geometry":
        """CSG Union.

        Args:
            other (Geometry): The other geometry.

        Returns:
            Geometry: The union of two geometries.

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval1 = ppsci.geometry.Interval(0, 1)
            >>> interval2 = ppsci.geometry.Interval(0.5, 1.5)
            >>> union = interval1.union(interval2)
            >>> union.bbox
            (array([[0.]]), array([[1.5]]))
            >>> rectangle1 = ppsci.geometry.Rectangle((0, 0), (2, 3))
            >>> rectangle2 = ppsci.geometry.Rectangle((0, 0), (3, 2))
            >>> union = rectangle1.union(rectangle2)
            >>> union.bbox
            (array([0., 0.], dtype=float32), array([3., 3.], dtype=float32))
            >>> cuboid1 = ppsci.geometry.Cuboid((0, 0, 0), (1, 2, 2))
            >>> cuboid2 = ppsci.geometry.Cuboid((0, 0, 0), (2, 1, 1))
            >>> union = cuboid1 | cuboid2
            >>> union.bbox
            (array([0., 0., 0.], dtype=float32), array([2., 2., 2.], dtype=float32))
        """
        from ppsci.geometry import csg

        return csg.CSGUnion(self, other)

    def __or__(self, other: "Geometry") -> "Geometry":
        """CSG Union.

        Args:
            other (Geometry): The other geometry.

        Returns:
            Geometry: The union of two geometries.

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval1 = ppsci.geometry.Interval(0, 1)
            >>> interval2 = ppsci.geometry.Interval(0.5, 1.5)
            >>> union = interval1.__or__(interval2)
            >>> union.bbox
            (array([[0.]]), array([[1.5]]))
            >>> rectangle1 = ppsci.geometry.Rectangle((0, 0), (2, 3))
            >>> rectangle2 = ppsci.geometry.Rectangle((0, 0), (3, 2))
            >>> union = rectangle1.__or__(rectangle2)
            >>> union.bbox
            (array([0., 0.], dtype=float32), array([3., 3.], dtype=float32))
            >>> cuboid1 = ppsci.geometry.Cuboid((0, 0, 0), (1, 2, 2))
            >>> cuboid2 = ppsci.geometry.Cuboid((0, 0, 0), (2, 1, 1))
            >>> union = cuboid1 | cuboid2
            >>> union.bbox
            (array([0., 0., 0.], dtype=float32), array([2., 2., 2.], dtype=float32))
        """
        from ppsci.geometry import csg

        return csg.CSGUnion(self, other)

    def difference(self, other: "Geometry") -> "Geometry":
        """CSG Difference.

        Args:
            other (Geometry): The other geometry.

        Returns:
            Geometry: The difference of two geometries.

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval1 = ppsci.geometry.Interval(0.0, 2.0)
            >>> interval2 = ppsci.geometry.Interval(1.0, 3.0)
            >>> difference = interval1.difference(interval2)
            >>> difference.bbox
            (array([[0.]]), array([[2.]]))
            >>> rectangle1 = ppsci.geometry.Rectangle((0.0, 0.0), (2.0, 3.0))
            >>> rectangle2 = ppsci.geometry.Rectangle((1.0, 1.0), (2.0, 2.0))
            >>> difference = rectangle1.difference(rectangle2)
            >>> difference.bbox
            (array([0., 0.], dtype=float32), array([2., 3.], dtype=float32))
            >>> cuboid1 = ppsci.geometry.Cuboid((0, 0, 0), (1, 2, 2))
            >>> cuboid2 = ppsci.geometry.Cuboid((0, 0, 0), (2, 1, 1))
            >>> difference = cuboid1 - cuboid2
            >>> difference.bbox
            (array([0., 0., 0.], dtype=float32), array([1., 2., 2.], dtype=float32))
        """
        from ppsci.geometry import csg

        return csg.CSGDifference(self, other)

    def __sub__(self, other: "Geometry") -> "Geometry":
        """CSG Difference.

        Args:
            other (Geometry): The other geometry.

        Returns:
            Geometry: The difference of two geometries.

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval1 = ppsci.geometry.Interval(0.0, 2.0)
            >>> interval2 = ppsci.geometry.Interval(1.0, 3.0)
            >>> difference = interval1.__sub__(interval2)
            >>> difference.bbox
            (array([[0.]]), array([[2.]]))
            >>> rectangle1 = ppsci.geometry.Rectangle((0.0, 0.0), (2.0, 3.0))
            >>> rectangle2 = ppsci.geometry.Rectangle((1.0, 1.0), (2.0, 2.0))
            >>> difference = rectangle1.__sub__(rectangle2)
            >>> difference.bbox
            (array([0., 0.], dtype=float32), array([2., 3.], dtype=float32))
            >>> cuboid1 = ppsci.geometry.Cuboid((0, 0, 0), (1, 2, 2))
            >>> cuboid2 = ppsci.geometry.Cuboid((0, 0, 0), (2, 1, 1))
            >>> difference = cuboid1 - cuboid2
            >>> difference.bbox
            (array([0., 0., 0.], dtype=float32), array([1., 2., 2.], dtype=float32))
        """
        from ppsci.geometry import csg

        return csg.CSGDifference(self, other)

    def intersection(self, other: "Geometry") -> "Geometry":
        """CSG Intersection.

        Args:
            other (Geometry): The other geometry.

        Returns:
            Geometry: The intersection of two geometries.

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval1 = ppsci.geometry.Interval(0.0, 1.0)
            >>> interval2 = ppsci.geometry.Interval(0.5, 1.5)
            >>> intersection = interval1.intersection(interval2)
            >>> intersection.bbox
            (array([[0.5]]), array([[1.]]))
            >>> rectangle1 = ppsci.geometry.Rectangle((0.0, 0.0), (2.0, 3.0))
            >>> rectangle2 = ppsci.geometry.Rectangle((0.0, 0.0), (3.0, 2.0))
            >>> intersection = rectangle1.intersection(rectangle2)
            >>> intersection.bbox
            (array([0., 0.], dtype=float32), array([2., 2.], dtype=float32))
            >>> cuboid1 = ppsci.geometry.Cuboid((0, 0, 0), (1, 2, 2))
            >>> cuboid2 = ppsci.geometry.Cuboid((0, 0, 0), (2, 1, 1))
            >>> intersection = cuboid1 & cuboid2
            >>> intersection.bbox
            (array([0., 0., 0.], dtype=float32), array([1., 1., 1.], dtype=float32))
        """
        from ppsci.geometry import csg

        return csg.CSGIntersection(self, other)

    def __and__(self, other: "Geometry") -> "Geometry":
        """CSG Intersection.

        Args:
            other (Geometry): The other geometry.

        Returns:
            Geometry: The intersection of two geometries.

        Examples:
            >>> import numpy as np
            >>> import ppsci
            >>> interval1 = ppsci.geometry.Interval(0.0, 1.0)
            >>> interval2 = ppsci.geometry.Interval(0.5, 1.5)
            >>> intersection = interval1.__and__(interval2)
            >>> intersection.bbox
            (array([[0.5]]), array([[1.]]))
            >>> rectangle1 = ppsci.geometry.Rectangle((0.0, 0.0), (2.0, 3.0))
            >>> rectangle2 = ppsci.geometry.Rectangle((0.0, 0.0), (3.0, 2.0))
            >>> intersection = rectangle1.__and__(rectangle2)
            >>> intersection.bbox
            (array([0., 0.], dtype=float32), array([2., 2.], dtype=float32))
            >>> cuboid1 = ppsci.geometry.Cuboid((0, 0, 0), (1, 2, 2))
            >>> cuboid2 = ppsci.geometry.Cuboid((0, 0, 0), (2, 1, 1))
            >>> intersection = cuboid1 & cuboid2
            >>> intersection.bbox
            (array([0., 0., 0.], dtype=float32), array([1., 1., 1.], dtype=float32))
        """
        from ppsci.geometry import csg

        return csg.CSGIntersection(self, other)

    def __str__(self) -> str:
        """Return the name of class.

        Returns:
            str: Meta information of geometry.

        Examples:
            >>> import ppsci
            >>> interval = ppsci.geometry.Interval(0, 1)
            >>> interval.__str__()
            "Interval, ndim = 1, bbox = (array([[0]]), array([[1]])), diam = 1, dim_keys = ('x',)"
            >>> rectangle = ppsci.geometry.Rectangle((0, 0), (1, 1))
            >>> rectangle.__str__()
            "Rectangle, ndim = 2, bbox = (array([0., 0.], dtype=float32), array([1., 1.], dtype=float32)), diam = 1.4142135381698608, dim_keys = ('x', 'y')"
            >>> cuboid = ppsci.geometry.Cuboid((0, 0, 0), (1, 1, 1))
            >>> cuboid.__str__()
            "Cuboid, ndim = 3, bbox = (array([0., 0., 0.], dtype=float32), array([1., 1., 1.], dtype=float32)), diam = 1.7320507764816284, dim_keys = ('x', 'y', 'z')"
        """
        return ", ".join(
            [
                self.__class__.__name__,
                f"ndim = {self.ndim}",
                f"bbox = {self.bbox}",
                f"diam = {self.diam}",
                f"dim_keys = {self.dim_keys}",
            ]
        )

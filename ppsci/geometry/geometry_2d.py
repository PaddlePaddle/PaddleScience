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

from typing import Tuple

import numpy as np
import paddle
from scipy import spatial

from ppsci.geometry import geometry
from ppsci.geometry import geometry_nd
from ppsci.geometry import sampler


class Disk(geometry.Geometry):
    """Class for disk geometry

    Args:
        center (Tuple[float, float]): Center point of disk [x0, y0].
        radius (float): Radius of disk.

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Disk((0.0, 0.0), 1.0)
    """

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = np.array(center, dtype=paddle.get_default_dtype())
        self.radius = radius
        super().__init__(2, (self.center - radius, self.center + radius), 2 * radius)

    def is_inside(self, x):
        return np.linalg.norm(x - self.center, axis=1) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center, axis=1), self.radius)

    def boundary_normal(self, x):
        ox = x - self.center
        ox_len = np.linalg.norm(ox, axis=1, keepdims=True)
        ox = (ox / ox_len) * np.isclose(ox_len, self.radius).astype(
            paddle.get_default_dtype()
        )
        return ox

    def random_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/DiskPointPicking.html
        rng = sampler.sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        return self.radius * np.stack((x, y), axis=1) + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(
            0, 2 * np.pi, num=n, endpoint=False, dtype=paddle.get_default_dtype()
        )
        X = np.stack((np.cos(theta), np.sin(theta)), axis=1)
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        theta = 2 * np.pi * sampler.sample(n, 1, random)
        X = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
        return self.radius * X + self.center

    def sdf_func(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance field.

        Args:
            points (np.ndarray): The coordinate points used to calculate the SDF value,
                the shape is [N, 2]

        Returns:
            np.ndarray: Unsquared SDF values of input points, the shape is [N, 1].

        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        if points.shape[1] != self.ndim:
            raise ValueError(
                f"Shape of given points should be [*, {self.ndim}], but got {points.shape}"
            )
        sdf = self.radius - np.linalg.norm(points - self.center, axis=1)
        sdf = -sdf[..., np.newaxis]
        return sdf


class Rectangle(geometry_nd.Hypercube):
    """Class for rectangle geometry

    Args:
        xmin (Tuple[float, float]): Bottom left corner point, [x0, y0].
        xmax (Tuple[float, float]): Top right corner point, [x1, y1].

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Rectangle((0.0, 0.0), (1.0, 1.0))
    """

    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)
        self.perimeter = 2 * np.sum(self.xmax - self.xmin)
        self.area = np.prod(self.xmax - self.xmin)

    def uniform_boundary_points(self, n):
        nx, ny = np.ceil(n / self.perimeter * (self.xmax - self.xmin)).astype(int)
        bottom = np.hstack(
            (
                np.linspace(
                    self.xmin[0],
                    self.xmax[0],
                    nx,
                    endpoint=False,
                    dtype=paddle.get_default_dtype(),
                ).reshape([nx, 1]),
                np.full([nx, 1], self.xmin[1], dtype=paddle.get_default_dtype()),
            )
        )
        right = np.hstack(
            (
                np.full([ny, 1], self.xmax[0], dtype=paddle.get_default_dtype()),
                np.linspace(
                    self.xmin[1],
                    self.xmax[1],
                    ny,
                    endpoint=False,
                    dtype=paddle.get_default_dtype(),
                ).reshape([ny, 1]),
            )
        )
        top = np.hstack(
            (
                np.linspace(
                    self.xmin[0], self.xmax[0], nx + 1, dtype=paddle.get_default_dtype()
                )[1:].reshape([nx, 1]),
                np.full([nx, 1], self.xmax[1], dtype=paddle.get_default_dtype()),
            )
        )
        left = np.hstack(
            (
                np.full([ny, 1], self.xmin[0], dtype=paddle.get_default_dtype()),
                np.linspace(
                    self.xmin[1], self.xmax[1], ny + 1, dtype=paddle.get_default_dtype()
                )[1:].reshape([ny, 1]),
            )
        )
        x = np.vstack((bottom, right, top, left))
        if len(x) > n:
            x = x[0:n]
        return x

    def random_boundary_points(self, n, random="pseudo"):
        l1 = self.xmax[0] - self.xmin[0]
        l2 = l1 + self.xmax[1] - self.xmin[1]
        l3 = l2 + l1
        u = np.ravel(sampler.sample(n + 10, 1, random))
        # Remove the possible points very close to the corners
        u = u[~np.isclose(u, l1 / self.perimeter)]
        u = u[~np.isclose(u, l3 / self.perimeter)]
        u = u[0:n]

        u *= self.perimeter
        x = []
        for l in u:
            if l < l1:
                x.append([self.xmin[0] + l, self.xmin[1]])
            elif l < l2:
                x.append([self.xmax[0], self.xmin[1] + (l - l1)])
            elif l < l3:
                x.append([self.xmax[0] - (l - l2), self.xmax[1]])
            else:
                x.append([self.xmin[0], self.xmax[1] - (l - l3)])
        return np.vstack(x)

    @staticmethod
    def is_valid(vertices):
        """Check if the geometry is a Rectangle."""
        return (
            len(vertices) == 4
            and np.isclose(np.prod(vertices[1] - vertices[0]), 0)
            and np.isclose(np.prod(vertices[2] - vertices[1]), 0)
            and np.isclose(np.prod(vertices[3] - vertices[2]), 0)
            and np.isclose(np.prod(vertices[0] - vertices[3]), 0)
        )

    def sdf_func(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance field.

        Args:
            points (np.ndarray): The coordinate points used to calculate the SDF value,
                the shape of the array is [N, 2].

        Returns:
            np.ndarray: Unsquared SDF values of input points, the shape is [N, 1].

        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        if points.shape[1] != self.ndim:
            raise ValueError(
                f"Shape of given points should be [*, {self.ndim}], but got {points.shape}"
            )
        center = (self.xmin + self.xmax) / 2
        dist_to_boundary = (
            np.abs(points - center) - np.array([self.xmax - self.xmin]) / 2
        )
        return (
            np.linalg.norm(np.maximum(dist_to_boundary, 0), axis=1)
            + np.minimum(np.max(dist_to_boundary, axis=1), 0)
        ).reshape(-1, 1)


class Triangle(geometry.Geometry):
    """Class for Triangle

    The order of vertices can be in a clockwise or counterclockwise direction. The
    vertices will be re-ordered in counterclockwise (right hand rule).

    Args:
        x1 (Tuple[float, float]): First point of Triangle [x0, y0].
        x2 (Tuple[float, float]): Second point of Triangle [x1, y1].
        x3 (Tuple[float, float]): Third point of Triangle [x2, y2].

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Triangle((0, 0), (1, 0), (0, 1))
    """

    def __init__(self, x1, x2, x3):
        self.area = polygon_signed_area([x1, x2, x3])
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            x2, x3 = x3, x2

        self.x1 = np.array(x1, dtype=paddle.get_default_dtype())
        self.x2 = np.array(x2, dtype=paddle.get_default_dtype())
        self.x3 = np.array(x3, dtype=paddle.get_default_dtype())

        self.v12 = self.x2 - self.x1
        self.v23 = self.x3 - self.x2
        self.v31 = self.x1 - self.x3
        self.l12 = np.linalg.norm(self.v12)
        self.l23 = np.linalg.norm(self.v23)
        self.l31 = np.linalg.norm(self.v31)
        self.n12 = self.v12 / self.l12
        self.n23 = self.v23 / self.l23
        self.n31 = self.v31 / self.l31
        self.n12_normal = clockwise_rotation_90(self.n12)
        self.n23_normal = clockwise_rotation_90(self.n23)
        self.n31_normal = clockwise_rotation_90(self.n31)
        self.perimeter = self.l12 + self.l23 + self.l31

        super().__init__(
            2,
            (np.minimum(x1, np.minimum(x2, x3)), np.maximum(x1, np.maximum(x2, x3))),
            self.l12
            * self.l23
            * self.l31
            / (
                self.perimeter
                * (self.l12 + self.l23 - self.l31)
                * (self.l23 + self.l31 - self.l12)
                * (self.l31 + self.l12 - self.l23)
            )
            ** 0.5,
        )

    def is_inside(self, x):
        # https://stackoverflow.com/a/2049593/12679294
        _sign = np.stack(
            [
                np.cross(self.v12, x - self.x1),
                np.cross(self.v23, x - self.x2),
                np.cross(self.v31, x - self.x3),
            ],
            axis=1,
        )
        return ~(np.any(_sign > 0, axis=-1) & np.any(_sign < 0, axis=-1))

    def on_boundary(self, x):
        l1 = np.linalg.norm(x - self.x1, axis=-1)
        l2 = np.linalg.norm(x - self.x2, axis=-1)
        l3 = np.linalg.norm(x - self.x3, axis=-1)
        return np.any(
            np.isclose(
                [l1 + l2 - self.l12, l2 + l3 - self.l23, l3 + l1 - self.l31],
                0,
                atol=1e-6,
            ),
            axis=0,
        )

    def boundary_normal(self, x):
        l1 = np.linalg.norm(x - self.x1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(x - self.x2, axis=-1, keepdims=True)
        l3 = np.linalg.norm(x - self.x3, axis=-1, keepdims=True)
        on12 = np.isclose(l1 + l2, self.l12)
        on23 = np.isclose(l2 + l3, self.l23)
        on31 = np.isclose(l3 + l1, self.l31)
        # Check points on the vertexes
        if np.any(np.count_nonzero(np.hstack([on12, on23, on31]), axis=-1) > 1):
            raise ValueError(
                "{}.boundary_normal do not accept points on the vertexes.".format(
                    self.__class__.__name__
                )
            )
        return self.n12_normal * on12 + self.n23_normal * on23 + self.n31_normal * on31

    def random_points(self, n, random="pseudo"):
        # There are two methods for triangle point picking.
        # Method 1 (used here):
        # - https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle
        # Method 2:
        # - http://mathworld.wolfram.com/TrianglePointPicking.html
        # - https://hbfs.wordpress.com/2010/10/05/random-points-in-a-triangle-generating-random-sequences-ii/
        # - https://stackoverflow.com/questions/19654251/random-point-inside-triangle-inside-java
        sqrt_r1 = np.sqrt(np.random.rand(n, 1))
        r2 = np.random.rand(n, 1)
        return (
            (1 - sqrt_r1) * self.x1
            + sqrt_r1 * (1 - r2) * self.x2
            + r2 * sqrt_r1 * self.x3
        )

    def uniform_boundary_points(self, n):
        density = n / self.perimeter
        x12 = (
            np.linspace(
                0,
                1,
                num=int(np.ceil(density * self.l12)),
                endpoint=False,
                dtype=paddle.get_default_dtype(),
            )[:, None]
            * self.v12
            + self.x1
        )
        x23 = (
            np.linspace(
                0,
                1,
                num=int(np.ceil(density * self.l23)),
                endpoint=False,
                dtype=paddle.get_default_dtype(),
            )[:, None]
            * self.v23
            + self.x2
        )
        x31 = (
            np.linspace(
                0,
                1,
                num=int(np.ceil(density * self.l31)),
                endpoint=False,
                dtype=paddle.get_default_dtype(),
            )[:, None]
            * self.v31
            + self.x3
        )
        x = np.vstack((x12, x23, x31))
        if len(x) > n:
            x = x[0:n]
        return x

    def random_boundary_points(self, n, random="pseudo"):
        u = np.ravel(sampler.sample(n + 2, 1, random))
        # Remove the possible points very close to the corners
        u = u[np.logical_not(np.isclose(u, self.l12 / self.perimeter))]
        u = u[np.logical_not(np.isclose(u, (self.l12 + self.l23) / self.perimeter))]
        u = u[:n]

        u *= self.perimeter
        x = []
        for l in u:
            if l < self.l12:
                x.append(l * self.n12 + self.x1)
            elif l < self.l12 + self.l23:
                x.append((l - self.l12) * self.n23 + self.x2)
            else:
                x.append((l - self.l12 - self.l23) * self.n31 + self.x3)
        return np.vstack(x)

    def sdf_func(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance field.

        Args:
            points (np.ndarray): The coordinate points used to calculate the SDF value,
                the shape of the array is [N, 2].

        Returns:
            np.ndarray: Unsquared SDF values of input points, the shape is [N, 1].

        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        if points.shape[1] != self.ndim:
            raise ValueError(
                f"Shape of given points should be [*, {self.ndim}], but got {points.shape}"
            )
        v1p = points - self.x1  # v1p: vector from x1 to points
        v2p = points - self.x2
        v3p = points - self.x3
        # vv12_p: vertical vector of points to v12(If the vertical point is in the extension of v12,
        # the vector will be the vector from x1 to points)
        vv12_p = (
            self.v12
            * np.clip(np.dot(v1p, self.v12.reshape(2, -1)) / self.l12**2, 0, 1)
            - v1p
        )
        vv23_p = (
            self.v23
            * np.clip(np.dot(v2p, self.v23.reshape(2, -1)) / self.l23**2, 0, 1)
            - v2p
        )
        vv31_p = (
            self.v31
            * np.clip(np.dot(v3p, self.v31.reshape(2, -1)) / self.l31**2, 0, 1)
            - v3p
        )
        is_inside = self.is_inside(points).reshape(-1, 1) * 2 - 1
        len_vv12_p = np.linalg.norm(vv12_p, axis=1, keepdims=True)
        len_vv23_p = np.linalg.norm(vv23_p, axis=1, keepdims=True)
        len_vv31_p = np.linalg.norm(vv31_p, axis=1, keepdims=True)
        mini_dist = np.minimum(np.minimum(len_vv12_p, len_vv23_p), len_vv31_p)
        return is_inside * mini_dist


class Polygon(geometry.Geometry):
    """Class for simple polygon.

    Args:
        vertices (Tuple[Tuple[float, float], ...]): The order of vertices can be in a
            clockwise or counterclockwisedirection. The vertices will be re-ordered in
            counterclockwise (right hand rule).

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Polygon(((0, 0), (1, 0), (2, 1), (2, 2), (0, 2)))
    """

    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=paddle.get_default_dtype())
        if len(vertices) == 3:
            raise ValueError("The polygon is a triangle. Use Triangle instead.")
        if Rectangle.is_valid(self.vertices):
            raise ValueError("The polygon is a rectangle. Use Rectangle instead.")

        self.area = polygon_signed_area(self.vertices)
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            self.vertices = np.flipud(self.vertices)

        self.diagonals = spatial.distance.squareform(
            spatial.distance.pdist(self.vertices)
        )
        super().__init__(
            2,
            (np.amin(self.vertices, axis=0), np.amax(self.vertices, axis=0)),
            np.max(self.diagonals),
        )
        self.nvertices = len(self.vertices)
        self.perimeter = np.sum(
            [self.diagonals[i, i + 1] for i in range(-1, self.nvertices - 1)]
        )
        self.bbox = np.array(
            [np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)],
            dtype=paddle.get_default_dtype(),
        )

        self.segments = self.vertices[1:] - self.vertices[:-1]
        self.segments = np.vstack((self.vertices[0] - self.vertices[-1], self.segments))
        self.normal = clockwise_rotation_90(self.segments.T).T
        self.normal = self.normal / np.linalg.norm(self.normal, axis=1).reshape(-1, 1)

    def is_inside(self, x):
        def wn_PnPoly(P, V):
            """Winding number algorithm.

            https://en.wikipedia.org/wiki/Point_in_polygon
            http://geomalgorithms.com/a03-_inclusion.html

            Args:
                P: A point.
                V: Vertex points of a polygon.

            Returns:
                wn: Winding number (=0 only if P is outside polygon).
            """
            wn = np.zeros(len(P))  # Winding number counter

            # Repeat the first vertex at end
            # Loop through all edges of the polygon
            for i in range(-1, self.nvertices - 1):  # Edge from V[i] to V[i+1]
                tmp = np.all(
                    np.hstack(
                        [
                            V[i, 1] <= P[:, 1:2],  # Start y <= P[1]
                            V[i + 1, 1] > P[:, 1:2],  # An upward crossing
                            is_left(V[i], V[i + 1], P) > 0,  # P left of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] += 1  # Have a valid up intersect
                tmp = np.all(
                    np.hstack(
                        [
                            V[i, 1] > P[:, 1:2],  # Start y > P[1]
                            V[i + 1, 1] <= P[:, 1:2],  # A downward crossing
                            is_left(V[i], V[i + 1], P) < 0,  # P right of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] -= 1  # Have a valid down intersect
            return wn

        return wn_PnPoly(x, self.vertices) != 0

    def on_boundary(self, x):
        _on = np.zeros(shape=len(x), dtype=np.int)
        for i in range(-1, self.nvertices - 1):
            l1 = np.linalg.norm(self.vertices[i] - x, axis=-1)
            l2 = np.linalg.norm(self.vertices[i + 1] - x, axis=-1)
            _on[np.isclose(l1 + l2, self.diagonals[i, i + 1])] += 1
        return _on > 0

    def random_points(self, n, random="pseudo"):
        x = np.empty((0, 2), dtype=paddle.get_default_dtype())
        vbbox = self.bbox[1] - self.bbox[0]
        while len(x) < n:
            x_new = sampler.sample(n, 2, "pseudo") * vbbox + self.bbox[0]
            x = np.vstack((x, x_new[self.is_inside(x_new)]))
        return x[:n]

    def uniform_boundary_points(self, n):
        density = n / self.perimeter
        x = []
        for i in range(-1, self.nvertices - 1):
            x.append(
                np.linspace(
                    0,
                    1,
                    num=int(np.ceil(density * self.diagonals[i, i + 1])),
                    endpoint=False,
                    dtype=paddle.get_default_dtype(),
                )[:, None]
                * (self.vertices[i + 1] - self.vertices[i])
                + self.vertices[i]
            )
        x = np.vstack(x)
        if len(x) > n:
            x = x[0:n]
        return x

    def random_boundary_points(self, n, random="pseudo"):
        u = np.ravel(sampler.sample(n + self.nvertices, 1, random))
        # Remove the possible points very close to the corners
        l = 0
        for i in range(0, self.nvertices - 1):
            l += self.diagonals[i, i + 1]
            u = u[np.logical_not(np.isclose(u, l / self.perimeter))]
        u = u[:n]
        u *= self.perimeter
        u.sort()

        x = []
        i = -1
        l0 = 0
        l1 = l0 + self.diagonals[i, i + 1]
        v = (self.vertices[i + 1] - self.vertices[i]) / self.diagonals[i, i + 1]
        for l in u:
            if l > l1:
                i += 1
                l0, l1 = l1, l1 + self.diagonals[i, i + 1]
                v = (self.vertices[i + 1] - self.vertices[i]) / self.diagonals[i, i + 1]
            x.append((l - l0) * v + self.vertices[i])
        return np.vstack(x)

    def sdf_func(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance field.
        Args:
            points (np.ndarray): The coordinate points used to calculate the SDF value,
                the shape is [N, 2]
        Returns:
            np.ndarray: Unsquared SDF values of input points, the shape is [N, 1].
        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        if points.shape[1] != self.ndim:
            raise ValueError(
                f"Shape of given points should be [*, {self.ndim}], but got {points.shape}"
            )
        sdf_value = np.empty((points.shape[0], 1), dtype=paddle.get_default_dtype())
        for n in range(points.shape[0]):
            distance = np.dot(
                points[n] - self.vertices[0], points[n] - self.vertices[0]
            )
            inside_tag = 1.0
            for i in range(self.vertices.shape[0]):
                j = (self.vertices.shape[0] - 1) if i == 0 else (i - 1)
                # Calculate the shortest distance from point P to each edge.
                vector_ij = self.vertices[j] - self.vertices[i]
                vector_in = points[n] - self.vertices[i]
                distance_vector = vector_in - vector_ij * np.clip(
                    np.dot(vector_in, vector_ij) / np.dot(vector_ij, vector_ij),
                    0.0,
                    1.0,
                )
                distance = np.minimum(
                    distance, np.dot(distance_vector, distance_vector)
                )
                # Calculate the inside and outside using the Odd-even rule
                odd_even_rule_number = np.array(
                    [
                        points[n][1] >= self.vertices[i][1],
                        points[n][1] < self.vertices[j][1],
                        vector_ij[0] * vector_in[1] > vector_ij[1] * vector_in[0],
                    ]
                )
                if odd_even_rule_number.all() or np.all(~odd_even_rule_number):
                    inside_tag *= -1.0
            sdf_value[n] = inside_tag * np.sqrt(distance)
        return -sdf_value


def polygon_signed_area(vertices):
    """The (signed) area of a simple polygon.

    If the vertices are in the counterclockwise direction, then the area is positive; if
    they are in the clockwise direction, the area is negative.

    Shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula

    Args:
        vertices (np.ndarray): Polygon vertices with shape of [N, 2].

    Returns:
        float: The (signed) area of a simple polygon.
    """
    x, y = zip(*vertices)
    x = np.array(list(x) + [x[0]], dtype=paddle.get_default_dtype())
    y = np.array(list(y) + [y[0]], dtype=paddle.get_default_dtype())
    return 0.5 * (np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]))


def clockwise_rotation_90(v):
    """Rotate a vector of 90 degrees clockwise about the origin.

    Args:
        v (np.ndarray): Vector with shape of [2, N].

    Returns:
        np.ndarray: Rotated vector with shape of [2, N].
    """
    return np.array([v[1], -v[0]], dtype=paddle.get_default_dtype())


def is_left(P0, P1, P2):
    """Test if a point is Left|On|Right of an infinite line.

    See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons".

    Args:
        P0 (np.ndarray): One point in the line.
        P1 (np.ndarray): One point in the line.
        P2 (np.ndarray): A array of point to be tested with shape of [N, 2].

    Returns:
        np.ndarray: >0 if P2 left of the line through P0 and P1, =0 if P2 on the line, <0 if P2
        right of the line.
    """
    return np.cross(P1 - P0, P2 - P0, axis=-1).reshape((-1, 1))

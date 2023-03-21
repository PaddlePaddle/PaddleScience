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

from __future__ import annotations

import warnings
from typing import Union

import numpy as np

try:
    import pymesh
except ImportError:
    warnings.warn(
        "Refer to README.md and install pymesh before using mesh API in neo_geometry"
    )

import pysdf

from .geometry import Geometry
from .inflation import pymesh_inflation
from .sampler import sample


class Mesh(Geometry):
    """A geometry represented by a point cloud, i.e., a set of points in space.

    Args:
        mesh(str, Mesh): Mesh file path, such as "/root/of/self.py_mesh.stl".
    """

    def __init__(self, mesh: Union[str, pymesh.Mesh]):
        if isinstance(mesh, str):
            self.py_mesh = pymesh.meshio.load_mesh(mesh)
        elif isinstance(mesh, pymesh.Mesh):
            self.py_mesh = mesh
        else:
            raise ValueError(
                f"type of mesh({type(mesh)} must be str or pymesh.Mesh")

        self.vertices = self.py_mesh.vertices
        self.faces = self.py_mesh.faces
        self.vectors = self.vertices[self.faces]
        super().__init__(
            self.vertices.shape[-1], (np.amin(
                self.vertices, axis=0), np.amax(
                    self.vertices, axis=0)),
            np.inf)
        self.v0 = self.vectors[:, 0]
        self.v1 = self.vectors[:, 1]
        self.v2 = self.vectors[:, 2]
        self.num_vertices = self.py_mesh.num_vertices
        self.num_faces = self.py_mesh.num_faces
        self.sdf = pysdf.SDF(self.vertices, self.faces)
        self.bounds = (
            ((np.min(self.vectors[:, :, 0])), np.max(self.vectors[:, :, 0])),
            ((np.min(self.vectors[:, :, 1])),
             np.max(self.vectors[:, :, 1])), ((np.min(self.vectors[:, :, 2])),
                                              np.max(self.vectors[:, :, 2])))

    def is_inside(self, x):
        # NOTE: point on boundary is included
        return self.sdf.contains(x)

    def on_boundary(self, x):
        x_sdf = self.sdf(x)
        return np.isclose(x_sdf, 0.0)

    def random_points(self, n, random="pseudo"):
        cur_n = 0
        all_points = []
        while cur_n < n:
            random_points = [
                sample(n, 1, random) * (e[1] - e[0]) + e[0]
                for e in self.bounds
            ]
            random_points = np.concatenate(random_points, axis=1)
            inner_mask = self.sdf.contains(random_points)
            valid_random_points = random_points[inner_mask]

            all_points.append(valid_random_points)
            cur_n += len(valid_random_points)

        all_points = np.concatenate(all_points, axis=0)
        if cur_n > n:
            all_points = all_points[:n]
        return all_points

    def uniform_boundary_points(self, n: int):
        """Compute the equispaced points on the boundary."""
        return self.sdf.sample_surface(n)

    def inflated_random_points(self, n, distance, random="pseudo"):
        if not isinstance(n, (tuple, list)):
            n = [n]
        if not isinstance(distance, (tuple, list)):
            distance = [distance]
        if len(n) != len(distance):
            raise ValueError(
                f"len(n)({len(n)}) must be equal to len(distance)({len(distance)})"
            )
        all_points = []
        for _n, _dist in zip(n, distance):
            inflated_mesh = Mesh(pymesh_inflation(self.py_mesh, _dist))
            cur_n = 0
            inflated_points = []
            while cur_n < _n:
                random_points = [
                    sample(_n, 1, random) * (e[1] - e[0]) + e[0]
                    for e in inflated_mesh.bounds
                ]
                random_points = np.concatenate(random_points, axis=1)
                inner_mask = inflated_mesh.sdf.contains(random_points)
                valid_random_points = random_points[inner_mask]

                inflated_points.append(valid_random_points)
                cur_n += len(valid_random_points)

            inflated_points = np.concatenate(inflated_points, axis=0)
            if cur_n > _n:
                inflated_points = inflated_points[:_n]
            all_points.append(inflated_points)

        return np.concatenate(all_points, axis=0)

    def inflated_random_boundary_points(self, n, distance, random="pseudo"):
        if not isinstance(n, (tuple, list)):
            n = [n]
        if not isinstance(distance, (tuple, list)):
            distance = [distance]
        if len(n) != len(distance):
            raise ValueError(
                f"len(n)({len(n)}) must be equal to len(distance)({len(distance)})"
            )
        all_points = []
        for _n, _dist in zip(n, distance):
            inflated_mesh = Mesh(pymesh_inflation(self.py_mesh, _dist))
            triangle_areas = area_of_triangles(
                inflated_mesh.v0, inflated_mesh.v1, inflated_mesh.v2)
            triangle_prob = triangle_areas / np.linalg.norm(
                triangle_areas, ord=1)
            triangle_index = np.arange(triangle_prob.shape[0])
            points_per_triangle = np.random.choice(
                triangle_index, _n, p=triangle_prob)
            points_per_triangle, _ = np.histogram(
                points_per_triangle,
                np.arange(triangle_prob.shape[0] + 1) - 0.5)

            inflated_boundary_points = []
            for index, num_point in enumerate(points_per_triangle):
                if num_point == 0:
                    continue
                sampled_points = sample_in_triangle(
                    inflated_mesh.v0[index], inflated_mesh.v1[index],
                    inflated_mesh.v2[index], num_point, random)
                inflated_boundary_points.append(sampled_points)
            inflated_boundary_points = np.concatenate(
                inflated_boundary_points, axis=0)
            all_points.append(inflated_boundary_points)

        return np.concatenate(all_points, axis=0)

    def random_boundary_points(self, n, random="pseudo"):
        triangle_areas = area_of_triangles(self.v0, self.v1, self.v2)
        triangle_prob = triangle_areas / np.linalg.norm(triangle_areas, ord=1)
        triangle_index = np.arange(triangle_prob.shape[0])
        points_per_triangle = np.random.choice(
            triangle_index, n, p=triangle_prob)
        points_per_triangle, _ = np.histogram(
            points_per_triangle, np.arange(triangle_prob.shape[0] + 1) - 0.5)

        all_points = []
        for index, num_point in enumerate(points_per_triangle):
            if num_point == 0:
                continue
            sampled_points = sample_in_triangle(self.v0[index], self.v1[index],
                                                self.v2[index], num_point,
                                                random)
            all_points.append(sampled_points)

        return np.concatenate(all_points, axis=0)

    def union(self, rhs: Mesh):
        csg = pymesh.CSGTree({
            "union": [{
                "mesh": self.py_mesh
            }, {
                "mesh": rhs.py_mesh
            }]
        })
        return Mesh(csg.mesh)

    def __or__(self, rhs):
        return self.union(rhs)

    def difference(self, rhs):
        csg = pymesh.CSGTree({
            "difference": [{
                "mesh": self.py_mesh
            }, {
                "mesh": rhs.py_mesh
            }]
        })
        return Mesh(csg.mesh)

    def __sub__(self, rhs):
        return self.difference(rhs)

    def intersection(self, rhs):
        csg = pymesh.CSGTree({
            "intersection": [{
                "mesh": self.py_mesh
            }, {
                "mesh": rhs.py_mesh
            }]
        })
        return Mesh(csg.mesh)

    def __and__(self, rhs):
        return self.intersection(rhs)


def area_of_triangles(v0, v1, v2):
    """ref https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle

    Args:
        v0 (np.ndarray): Coordinates of the first vertex of the triangle surface with shape of [N, 3].
        v1 (np.ndarray): Coordinates of the second vertex of the triangle surface with shape of [N, 3].
        v2 (np.ndarray): Coordinates of the third vertex of the triangle surface with shape of [N, 3].

    Returns:
        np.ndarray: Area of each triangle with shape of [N, ].
    """
    a = np.sqrt((v0[:, 0] - v1[:, 0])**2 + (v0[:, 1] - v1[:, 1])**2 + (
        v0[:, 2] - v1[:, 2])**2 + 1e-10)
    b = np.sqrt((v1[:, 0] - v2[:, 0])**2 + (v1[:, 1] - v2[:, 1])**2 + (
        v1[:, 2] - v2[:, 2])**2 + 1e-10)
    c = np.sqrt((v0[:, 0] - v2[:, 0])**2 + (v0[:, 1] - v2[:, 1])**2 + (
        v0[:, 2] - v2[:, 2])**2 + 1e-10)
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c) + 1e-10)
    return area


def sample_in_triangle(v0, v1, v2, n, random="pseudo"):
    """
    Uniformly sample n points in an 3D triangle defined by 3 vertices v0, v1, v2
    https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle

    Args:
        v0 (np.ndarray): Coordinates of the first vertex of an triangle with shape of [3, ].
        v1 (np.ndarray): Coordinates of the second vertex of an triangle with shape of [3, ].
        v2 (np.ndarray): Coordinates of the third vertex of an triangle with shape of [3, ].
        n (int): Number of points to be sampled.

    Returns:
        np.ndarray: Coordinates of sampled n points.
    """
    r1 = sample(n, 1, random).flatten()
    r2 = sample(n, 1, random).flatten()
    s1 = np.sqrt(r1)
    x = v0[0] * (1.0 - s1) + v1[0] * (1.0 - r2) * s1 + v2[0] * r2 * s1
    y = v0[1] * (1.0 - s1) + v1[1] * (1.0 - r2) * s1 + v2[1] * r2 * s1
    z = v0[2] * (1.0 - s1) + v1[2] * (1.0 - r2) * s1 + v2[2] * r2 * s1
    return np.stack([x, y, z], axis=1)

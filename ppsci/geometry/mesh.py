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

from __future__ import annotations

from typing import Union

import numpy as np
import pysdf

from ppsci.geometry import geometry
from ppsci.geometry import geometry_3d
from ppsci.geometry import sampler
from ppsci.utils import checker
from ppsci.utils import misc


class Mesh(geometry.Geometry):
    """Class for mesh geometry.

    Args:
        mesh (Union[str, Mesh]): Mesh file path or mesh object, such as "/path/to/mesh.stl".

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Mesh("/path/to/mesh.stl")  # doctest: +SKIP
    """

    def __init__(self, mesh: Union[Mesh, str]):
        # check if pymesh is installed when using Mesh Class
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ModuleNotFoundError

        if isinstance(mesh, str):
            self.py_mesh = pymesh.meshio.load_mesh(mesh)
        elif isinstance(mesh, pymesh.Mesh):
            self.py_mesh = mesh
        else:
            raise ValueError(f"type of mesh({type(mesh)} should be str or pymesh.Mesh")

        self.init_mesh()

    def init_mesh(self):
        """Initialize necessary variables for mesh"""
        if "face_normal" not in self.py_mesh.get_attribute_names():
            self.py_mesh.add_attribute("face_normal")
        self.face_normal = self.py_mesh.get_attribute("face_normal").reshape([-1, 3])

        if "face_area" not in self.py_mesh.get_attribute_names():
            self.py_mesh.add_attribute("face_area")
        self.face_area = self.py_mesh.get_attribute("face_area").reshape([-1])

        self.vertices = self.py_mesh.vertices
        self.faces = self.py_mesh.faces
        self.vectors = self.vertices[self.faces]
        super().__init__(
            self.vertices.shape[-1],
            (np.amin(self.vertices, axis=0), np.amax(self.vertices, axis=0)),
            np.inf,
        )
        self.v0 = self.vectors[:, 0]
        self.v1 = self.vectors[:, 1]
        self.v2 = self.vectors[:, 2]
        self.num_vertices = self.py_mesh.num_vertices
        self.num_faces = self.py_mesh.num_faces
        self.sdf = pysdf.SDF(self.vertices, self.faces)
        self.bounds = (
            ((np.min(self.vectors[:, :, 0])), np.max(self.vectors[:, :, 0])),
            ((np.min(self.vectors[:, :, 1])), np.max(self.vectors[:, :, 1])),
            ((np.min(self.vectors[:, :, 2])), np.max(self.vectors[:, :, 2])),
        )

    def is_inside(self, x):
        # NOTE: point on boundary is included
        return self.sdf.contains(x)

    def on_boundary(self, x):
        x_sdf = self.sdf(x)
        return np.isclose(x_sdf, 0.0)

    def translate(self, translation, relative=True):
        vertices = np.array(self.vertices)
        faces = np.array(self.faces)

        # check if open3d is installed before using inflation
        if not checker.dynamic_import_to_globals(["open3d"]):
            raise ModuleNotFoundError

        open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(faces),
        )
        open3d_mesh = open3d_mesh.translate(translation, relative)
        self.py_mesh = pymesh.form_mesh(
            np.asarray(open3d_mesh.vertices, dtype="float32"), faces
        )
        self.init_mesh()
        return self

    def scale(self, scale, center=(0, 0, 0)):
        vertices = np.array(self.vertices)
        faces = np.array(self.faces)

        # check if open3d is installed before using inflation
        if not checker.dynamic_import_to_globals(["open3d"]):
            raise ModuleNotFoundError

        open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(faces),
        )
        open3d_mesh.scale(scale, center)
        self.py_mesh = pymesh.form_mesh(
            np.asarray(open3d_mesh.vertices, dtype="float32"), faces
        )
        self.init_mesh()
        return self

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
                f"len(n)({len(n)}) should be equal to len(distance)({len(distance)})"
            )

        # check if open3d is installed before using inflation
        if not checker.dynamic_import_to_globals(["open3d"]):
            raise ModuleNotFoundError
        from ppsci.geometry import inflation

        all_points = []
        for _n, _dist in zip(n, distance):
            inflated_mesh = Mesh(inflation.pymesh_inflation(self.py_mesh, _dist))
            cur_n = 0
            inflated_points = []
            while cur_n < _n:
                random_points = [
                    sampler.sample(_n, 1, random) * (e[1] - e[0]) + e[0]
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
                f"len(n)({len(n)}) should be equal to len(distance)({len(distance)})"
            )
        all_points = []
        all_normal = []
        all_area = []

        # check if open3d is installed before using inflation module
        if not checker.dynamic_import_to_globals(["open3d"]):
            raise ModuleNotFoundError
        from ppsci.geometry import inflation

        for _n, _dist in zip(n, distance):
            inflated_mesh = Mesh(inflation.pymesh_inflation(self.py_mesh, _dist))
            triangle_areas = area_of_triangles(
                inflated_mesh.v0, inflated_mesh.v1, inflated_mesh.v2
            )
            triangle_prob = triangle_areas / np.linalg.norm(triangle_areas, 1)
            triangle_index = np.arange(triangle_prob.shape[0])
            points_per_triangle = np.random.choice(triangle_index, _n, p=triangle_prob)
            points_per_triangle, _ = np.histogram(
                points_per_triangle, np.arange(triangle_prob.shape[0] + 1) - 0.5
            )

            all_points_n = []
            all_normal_n = []
            all_area_n = []
            for index, nr_p in enumerate(points_per_triangle):
                if nr_p == 0:
                    continue
                sampled_points = sample_in_triangle(
                    inflated_mesh.v0[index],
                    inflated_mesh.v1[index],
                    inflated_mesh.v2[index],
                    nr_p,
                    random,
                )
                normal = np.tile(inflated_mesh.face_normal[index], [nr_p, 1])
                area = np.full([nr_p, 1], triangle_areas[index] / nr_p)

                all_points_n.append(sampled_points)
                all_normal_n.append(normal)
                all_area_n.append(area)

            all_points_n = np.concatenate(all_points_n, axis=0, dtype="float32")
            all_normal_n = np.concatenate(all_normal_n, axis=0, dtype="float32")
            all_area_n = np.concatenate(all_area_n, axis=0, dtype="float32")
            all_area_n = np.full_like(all_area_n, all_area_n.sum() / _n)

            all_points.append(all_points_n)
            all_normal.append(all_normal_n)
            all_area.append(all_area_n)

        all_points = np.concatenate(all_points, axis=0, dtype="float32")
        all_normal = np.concatenate(all_normal, axis=0, dtype="float32")
        all_area = np.concatenate(all_area, axis=0, dtype="float32")
        return all_points, all_normal, all_area

    def _approximate_area(self, n_appr, random="pseudo") -> float:
        triangle_areas = area_of_triangles(self.v0, self.v1, self.v2)
        triangle_prob = triangle_areas / np.linalg.norm(triangle_areas, ord=1)
        triangle_index = np.arange(triangle_prob.shape[0])
        points_per_triangle = np.random.choice(triangle_index, n_appr, p=triangle_prob)
        points_per_triangle, _ = np.histogram(
            points_per_triangle, np.arange(triangle_prob.shape[0] + 1) - 0.5
        )

        all_area = []
        for index, nr_p in enumerate(points_per_triangle):
            if nr_p == 0:
                continue
            area = np.full([nr_p, 1], triangle_areas[index] / nr_p)
            all_area.append(area)

        all_area = np.concatenate(all_area, axis=0, dtype="float32")
        return all_area.sum()

    def random_boundary_points(self, n, random="pseudo"):
        triangle_areas = area_of_triangles(self.v0, self.v1, self.v2)
        triangle_prob = triangle_areas / np.linalg.norm(triangle_areas, ord=1)
        triangle_index = np.arange(triangle_prob.shape[0])
        points_per_triangle = np.random.choice(triangle_index, n, p=triangle_prob)
        points_per_triangle, _ = np.histogram(
            points_per_triangle, np.arange(triangle_prob.shape[0] + 1) - 0.5
        )

        all_points = []
        all_normal = []
        all_area = []
        for index, nr_p in enumerate(points_per_triangle):
            if nr_p == 0:
                continue
            sampled_points = sample_in_triangle(
                self.v0[index], self.v1[index], self.v2[index], nr_p, random
            )
            normal = np.tile(self.face_normal[index], [nr_p, 1])
            area = np.full([nr_p, 1], triangle_areas[index] / nr_p)

            all_points.append(sampled_points)
            all_normal.append(normal)
            all_area.append(area)

        all_points = np.concatenate(all_points, axis=0, dtype="float32")
        all_normal = np.concatenate(all_normal, axis=0, dtype="float32")
        all_area = np.concatenate(all_area, axis=0, dtype="float32")
        all_area = np.full_like(all_area, all_area.sum() / n)
        return all_points, all_normal, all_area

    def sample_boundary(
        self, n, random="pseudo", criteria=None, evenly=False, inflation_dist=None
    ):
        # TODO(sensen): support for time-dependent points(repeat data in time)
        if inflation_dist is not None:
            x, normal, area = self.inflated_random_boundary_points(
                n, inflation_dist, random
            )
        else:
            x = np.empty(shape=(n, self.ndim), dtype="float32")
            _size, _ntry, _nsuc = 0, 0, 0
            while _size < n:
                if evenly:
                    raise ValueError(
                        "Can't sample evenly on mesh now, please set evenly=False."
                    )
                    # points, normal, area = self.uniform_boundary_points(n, False)
                else:
                    points, normal, area = self.random_boundary_points(n, random)

                if criteria is not None:
                    criteria_mask = criteria(*np.split(points, self.ndim, 1)).flatten()
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
                        "please check correctness of geometry and given creteria."
                    )

        normal_dict = misc.convert_to_dict(
            normal, [f"normal_{key}" for key in self.dim_keys if key != "t"]
        )
        area_dict = misc.convert_to_dict(area, ["area"])
        x_dict = misc.convert_to_dict(x, self.dim_keys)
        return {**x_dict, **normal_dict, **area_dict}

    def random_points(self, n, random="pseudo"):
        cur_n = 0
        all_points = []
        cuboid = geometry_3d.Cuboid(
            [bound[0] for bound in self.bounds],
            [bound[1] for bound in self.bounds],
        )
        while cur_n < n:
            random_points = cuboid.random_points(n, random)
            inner_mask = self.sdf.contains(random_points)
            valid_random_points = random_points[inner_mask]

            all_points.append(valid_random_points)
            cur_n += len(valid_random_points)

        all_points = np.concatenate(all_points, axis=0)
        if cur_n > n:
            all_points = all_points[:n]

        return all_points

    def sample_interior(self, n, random="pseudo", criteria=None, evenly=False):
        """Sample random points in the geometry and return those meet criteria."""
        x = np.empty(shape=(n, self.ndim), dtype="float32")
        _size, _ntry, _nsuc = 0, 0, 0
        while _size < n:
            if evenly:
                # TODO(sensen): implement uniform sample for mesh interior.
                raise NotImplementedError(
                    "uniformly sample for interior in mesh is not support yet"
                )
                points = self.uniform_points(n)
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
                    "please check correctness of geometry and given creteria."
                )

        x_dict = misc.convert_to_dict(x, self.dim_keys)

        volume = np.prod([bound[1] - bound[0] for bound in self.bounds])
        area = np.full((n, 1), volume / n, "float32")
        area_dict = misc.convert_to_dict(area, ["area"])
        return {**x_dict, **area_dict}

    def union(self, rhs):
        csg = pymesh.CSGTree({"union": [{"mesh": self.py_mesh}, {"mesh": rhs.py_mesh}]})
        return Mesh(csg.mesh)

    def __or__(self, rhs):
        return self.union(rhs)

    def difference(self, rhs):
        csg = pymesh.CSGTree(
            {"difference": [{"mesh": self.py_mesh}, {"mesh": rhs.py_mesh}]}
        )
        return Mesh(csg.mesh)

    def __sub__(self, rhs):
        return self.difference(rhs)

    def intersection(self, rhs):
        csg = pymesh.CSGTree(
            {"intersection": [{"mesh": self.py_mesh}, {"mesh": rhs.py_mesh}]}
        )
        return Mesh(csg.mesh)

    def __and__(self, rhs):
        return self.intersection(rhs)

    def __str__(self) -> str:
        """Return the name of class"""
        return ", ".join(
            [
                self.__class__.__name__,
                f"num_vertices = {self.num_vertices}",
                f"num_faces = {self.num_faces}",
                f"bounds = {self.bounds}",
                f"dim_keys = {self.dim_keys}",
            ]
        )


def area_of_triangles(v0, v1, v2):
    """ref https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle

    Args:
        v0 (np.ndarray): Coordinates of the first vertex of the triangle surface with shape of [N, 3].
        v1 (np.ndarray): Coordinates of the second vertex of the triangle surface with shape of [N, 3].
        v2 (np.ndarray): Coordinates of the third vertex of the triangle surface with shape of [N, 3].

    Returns:
        np.ndarray: Area of each triangle with shape of [N, ].
    """
    a = np.sqrt(
        (v0[:, 0] - v1[:, 0]) ** 2
        + (v0[:, 1] - v1[:, 1]) ** 2
        + (v0[:, 2] - v1[:, 2]) ** 2
        + 1e-10
    )
    b = np.sqrt(
        (v1[:, 0] - v2[:, 0]) ** 2
        + (v1[:, 1] - v2[:, 1]) ** 2
        + (v1[:, 2] - v2[:, 2]) ** 2
        + 1e-10
    )
    c = np.sqrt(
        (v0[:, 0] - v2[:, 0]) ** 2
        + (v0[:, 1] - v2[:, 1]) ** 2
        + (v0[:, 2] - v2[:, 2]) ** 2
        + 1e-10
    )
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
    r1 = sampler.sample(n, 1, random).flatten()
    r2 = sampler.sample(n, 1, random).flatten()
    s1 = np.sqrt(r1)
    x = v0[0] * (1.0 - s1) + v1[0] * (1.0 - r2) * s1 + v2[0] * r2 * s1
    y = v0[1] * (1.0 - s1) + v1[1] * (1.0 - r2) * s1 + v2[1] * r2 * s1
    z = v0[2] * (1.0 - s1) + v1[2] * (1.0 - r2) * s1 + v2[2] * r2 * s1
    return np.stack([x, y, z], axis=1)

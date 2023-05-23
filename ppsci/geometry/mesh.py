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

from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import paddle
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
        import pymesh

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

        self.open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(np.array(self.py_mesh.vertices)),
            open3d.utility.Vector3iVector(np.array(self.py_mesh.faces)),
        )
        self.open3d_mesh.compute_vertex_normals()

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
        self.pysdf = pysdf.SDF(self.vertices, self.faces)
        self.bounds = (
            ((np.min(self.vectors[:, :, 0])), np.max(self.vectors[:, :, 0])),
            ((np.min(self.vectors[:, :, 1])), np.max(self.vectors[:, :, 1])),
            ((np.min(self.vectors[:, :, 2])), np.max(self.vectors[:, :, 2])),
        )

    def sdf_func(self, points: np.ndarray) -> np.ndarray:
        """Compute signed distance field.

        Args:
            points (np.ndarray): The coordinate points used to calculate the SDF value,
                the shape is [N, 3]

        Returns:
            np.ndarray: Unsquared SDF values of input points, the shape is [N, 1].

        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        sdf, _, _, _ = pymesh.signed_distance_to_mesh(self.py_mesh, points)
        sdf = sdf[..., np.newaxis]
        return sdf

    def is_inside(self, x):
        # NOTE: point on boundary is included
        return self.pysdf.contains(x)

    def on_boundary(self, x):
        return np.isclose(self.sdf_func(x), 0.0).flatten()

    def translate(self, translation, relative=True):
        vertices = np.array(self.vertices, dtype=paddle.get_default_dtype())
        faces = np.array(self.faces)

        if not checker.dynamic_import_to_globals(["open3d", "pymesh"]):
            raise ModuleNotFoundError
        import open3d
        import pymesh

        open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(faces),
        )
        open3d_mesh = open3d_mesh.translate(translation, relative)
        self.py_mesh = pymesh.form_mesh(
            np.asarray(open3d_mesh.vertices, dtype=paddle.get_default_dtype()), faces
        )
        self.init_mesh()
        return self

    def scale(self, scale, center=(0, 0, 0)):
        vertices = np.array(self.vertices, dtype=paddle.get_default_dtype())
        faces = np.array(self.faces, dtype=paddle.get_default_dtype())

        if not checker.dynamic_import_to_globals(["open3d", "pymesh"]):
            raise ModuleNotFoundError
        import open3d
        import pymesh

        open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(faces),
        )
        open3d_mesh.scale(scale, center)
        self.py_mesh = pymesh.form_mesh(
            np.asarray(open3d_mesh.vertices, dtype=paddle.get_default_dtype()), faces
        )
        self.init_mesh()
        return self

    def uniform_boundary_points(self, n: int):
        """Compute the equispaced points on the boundary."""
        return self.pysdf.sample_surface(n)

    def inflated_random_points(self, n, distance, random="pseudo"):
        if not isinstance(n, (tuple, list)):
            n = [n]
        if not isinstance(distance, (tuple, list)):
            distance = [distance]
        if len(n) != len(distance):
            raise ValueError(
                f"len(n)({len(n)}) should be equal to len(distance)({len(distance)})"
            )

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
                inner_mask = inflated_mesh.pysdf.contains(random_points)
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

            all_points_n = np.concatenate(
                all_points_n, axis=0, dtype=paddle.get_default_dtype()
            )
            all_normal_n = np.concatenate(
                all_normal_n, axis=0, dtype=paddle.get_default_dtype()
            )
            all_area_n = np.concatenate(
                all_area_n, axis=0, dtype=paddle.get_default_dtype()
            )
            all_area_n = np.full_like(all_area_n, all_area_n.sum() / _n)

            all_points.append(all_points_n)
            all_normal.append(all_normal_n)
            all_area.append(all_area_n)

        all_points = np.concatenate(
            all_points, axis=0, dtype=paddle.get_default_dtype()
        )
        all_normal = np.concatenate(
            all_normal, axis=0, dtype=paddle.get_default_dtype()
        )
        all_area = np.concatenate(all_area, axis=0, dtype=paddle.get_default_dtype())
        return all_points, all_normal, all_area

    def _approximate_area(
        self,
        random: str = "pseudo",
        criteria: Optional[Callable] = None,
        n_appr: int = 20000,
    ) -> float:
        """Approximate area with given `criteria` using `n_appr` points.

        Args:
            n_appr (int): Number of points for approximating area.
            criteria (Callable): Criteria function.
            random (str, optional): Random method. Defaults to "pseudo".

        Returns:
            float: Approximated area.
        """
        areas = []
        for i in range(self.num_faces):
            sampled_points = sample_in_triangle(
                self.v0[i], self.v1[i], self.v2[i], n_appr, random
            )
            if criteria is not None:
                criteria_mask = criteria(
                    *np.split(sampled_points, self.ndim, 1)
                ).flatten()
            else:
                criteria_mask = np.full((n_appr,), True)
            valid_area = (criteria_mask.sum() / n_appr) * self.face_area[i]
            areas.append(valid_area)

        return np.asarray(areas, paddle.get_default_dtype())

    # def precise_on_boundary(self, points: np.ndarray, normals: np.ndarray):
    #     """judge whether points is accurately on boundary.

    #     Args:
    #         points (np.ndarray): Points.
    #         normals (np.ndarray): Normals for each points.

    #     Returns:
    #         np.ndarray: If on boundary, true for yes, false for not.
    #     """
    #     EPS = 1e-6
    #     points_pos_normals = points + normals * EPS
    #     points_neg_normals = points - normals * EPS
    #     pos_sdf = self.sdf_func(points_pos_normals)
    #     neg_sdf = self.sdf_func(points_neg_normals)
    #     return (pos_sdf * neg_sdf <= 0)[:, 0]

    def random_boundary_points(self, n, random="pseudo", criteria=None):
        triangle_areas = self._approximate_area(random, criteria)
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
                self.v0[index], self.v1[index], self.v2[index], nr_p, random, criteria
            )
            normal = np.tile(self.face_normal[index], [nr_p, 1])
            area = np.full(
                [nr_p, 1],
                triangle_areas[index] / nr_p,
                dtype=paddle.get_default_dtype(),
            )

            all_points.append(sampled_points)
            all_normal.append(normal)
            all_area.append(area)

        all_points = np.concatenate(
            all_points, axis=0, dtype=paddle.get_default_dtype()
        )
        all_normal = np.concatenate(
            all_normal, axis=0, dtype=paddle.get_default_dtype()
        )
        all_area = np.concatenate(all_area, axis=0, dtype=paddle.get_default_dtype())
        all_area = np.full_like(all_area, all_area.sum() / n)
        return all_points, all_normal, all_area

    def sample_boundary(
        self, n, random="pseudo", criteria=None, evenly=False, inflation_dist=None
    ):
        # TODO(sensen): support for time-dependent points(repeat data in time)
        if inflation_dist is not None:
            points, normals, areas = self.inflated_random_boundary_points(
                n, inflation_dist, random
            )
        else:
            if evenly:
                raise ValueError(
                    "Can't sample evenly on mesh now, please set evenly=False."
                )
                # points, normal, area = self.uniform_boundary_points(n, False)
            else:
                points, normals, areas = self.random_boundary_points(
                    n, random, criteria
                )

        x_dict = misc.convert_to_dict(points, self.dim_keys)
        normal_dict = misc.convert_to_dict(
            normals, [f"normal_{key}" for key in self.dim_keys if key != "t"]
        )
        area_dict = misc.convert_to_dict(areas, ["area"])
        return {**x_dict, **normal_dict, **area_dict}

    def random_points(self, n, random="pseudo", criteria=None):
        _size = 0
        all_points = []
        cuboid = geometry_3d.Cuboid(
            [bound[0] for bound in self.bounds],
            [bound[1] for bound in self.bounds],
        )
        _ntry, _nsuc = 0, 0
        while _size < n:
            random_points = cuboid.random_points(n, random)
            valid_mask = self.is_inside(random_points)
            if criteria:
                valid_mask &= criteria(
                    *np.split(random_points, self.ndim, axis=1)
                ).flatten()
            valid_random_points = random_points[valid_mask]
            _nsuc += len(valid_random_points)

            if len(valid_random_points) > n - _size:
                valid_random_points = valid_random_points[: n - _size]

            all_points.append(valid_random_points)
            _size += len(valid_random_points)
            _ntry += n

        all_points = np.concatenate(all_points, axis=0)
        all_areas = np.full(
            (n, 1), np.prod([b[1] - b[0] for b in self.bounds]) * (_nsuc / _ntry) / n
        )
        return all_points, all_areas

    def sample_interior(self, n, random="pseudo", criteria=None, evenly=False):
        """Sample random points in the geometry and return those meet criteria."""
        if evenly:
            # TODO(sensen): implement uniform sample for mesh interior.
            raise NotImplementedError(
                "uniformly sample for interior in mesh is not support yet"
            )
            # points, area = self.uniform_points(n)
        else:
            points, area = self.random_points(n, random, criteria)

        x_dict = misc.convert_to_dict(points, self.dim_keys)
        area_dict = misc.convert_to_dict(area, ["area"])

        # NOTE: add negtive to the sdf values for positive weight.
        sdf = -self.sdf_func(points)
        sdf_dict = misc.convert_to_dict(sdf, ["sdf"])

        return {**x_dict, **area_dict, **sdf_dict}

    def union(self, rhs):
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ModuleNotFoundError
        import pymesh

        csg = pymesh.CSGTree({"union": [{"mesh": self.py_mesh}, {"mesh": rhs.py_mesh}]})
        return Mesh(csg.mesh)

    def __or__(self, rhs):
        return self.union(rhs)

    def __add__(self, rhs):
        return self.union(rhs)

    def difference(self, rhs):
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ModuleNotFoundError
        import pymesh

        csg = pymesh.CSGTree(
            {"difference": [{"mesh": self.py_mesh}, {"mesh": rhs.py_mesh}]}
        )
        return Mesh(csg.mesh)

    def __sub__(self, rhs):
        return self.difference(rhs)

    def intersection(self, rhs):
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ModuleNotFoundError
        import pymesh

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
    p = (a + b + c) / 2
    area = np.sqrt(p * (p - a) * (p - b) * (p - c) + 1e-10)
    return area


def sample_in_triangle(v0, v1, v2, n, random="pseudo", criteria=None):
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
    all_x, all_y, all_z = [], [], []
    _size = 0
    while _size < n:
        r1 = sampler.sample(n, 1, random).flatten()
        r2 = sampler.sample(n, 1, random).flatten()
        s1 = np.sqrt(r1)
        x = v0[0] * (1.0 - s1) + v1[0] * (1.0 - r2) * s1 + v2[0] * r2 * s1
        y = v0[1] * (1.0 - s1) + v1[1] * (1.0 - r2) * s1 + v2[1] * r2 * s1
        z = v0[2] * (1.0 - s1) + v1[2] * (1.0 - r2) * s1 + v2[2] * r2 * s1

        if criteria is not None:
            criteria_mask = criteria(x, y, z).flatten()
            x = x[criteria_mask]
            y = y[criteria_mask]
            z = z[criteria_mask]

        if len(x) > n - _size:
            x = x[: n - _size]
            y = y[: n - _size]
            z = z[: n - _size]

        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        _size += len(x)

    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_z = np.concatenate(all_z, axis=0)

    return np.stack([all_x, all_y, all_z], axis=1)

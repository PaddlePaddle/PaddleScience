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

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle

from ppsci.geometry import geometry
from ppsci.geometry import geometry_3d
from ppsci.geometry import sampler
from ppsci.utils import checker
from ppsci.utils import misc

if TYPE_CHECKING:
    import pymesh


class Mesh(geometry.Geometry):
    """Class for mesh geometry.

    Args:
        mesh (Union[str, Mesh]): Mesh file path or mesh object, such as "/path/to/mesh.stl".

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.Mesh("/path/to/mesh.stl")  # doctest: +SKIP
    """

    def __init__(self, mesh: Union["pymesh.Mesh", str]):
        # check if pymesh is installed when using Mesh Class
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ImportError(
                "Could not import pymesh python package."
                "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import pymesh

        if isinstance(mesh, str):
            self.py_mesh = pymesh.meshio.load_mesh(mesh)
        elif isinstance(mesh, pymesh.Mesh):
            self.py_mesh = mesh
        else:
            raise ValueError("arg `mesh` should be path string or `pymesh.Mesh`")

        self.init_mesh()

    @classmethod
    def from_pymesh(cls, mesh: "pymesh.Mesh") -> "Mesh":
        """Instantiate Mesh object with given PyMesh object.

        Args:
            mesh (pymesh.Mesh): PyMesh object.

        Returns:
            Mesh: Instantiated ppsci.geometry.Mesh object.

        Examples:
            >>> import ppsci
            >>> import pymesh  # doctest: +SKIP
            >>> import numpy as np  # doctest: +SKIP
            >>> box = pymesh.generate_box_mesh(np.array([0, 0, 0]), np.array([1, 1, 1]))  # doctest: +SKIP
            >>> mesh = ppsci.geometry.Mesh.from_pymesh(box)  # doctest: +SKIP
            >>> print(mesh.vertices)  # doctest: +SKIP
            [[0. 0. 0.]
             [1. 0. 0.]
             [1. 1. 0.]
             [0. 1. 0.]
             [0. 0. 1.]
             [1. 0. 1.]
             [1. 1. 1.]
             [0. 1. 1.]]
        """
        # check if pymesh is installed when using Mesh Class
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ImportError(
                "Could not import pymesh python package."
                "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import pymesh

        if isinstance(mesh, pymesh.Mesh):
            return cls(mesh)
        else:
            raise ValueError(
                f"arg `mesh` should be type of `pymesh.Mesh`, but got {type(mesh)}"
            )

    def init_mesh(self):
        """Initialize necessary variables for mesh"""
        if "face_normal" not in self.py_mesh.get_attribute_names():
            self.py_mesh.add_attribute("face_normal")
        self.face_normal = self.py_mesh.get_attribute("face_normal").reshape([-1, 3])

        if not checker.dynamic_import_to_globals(["open3d"]):
            raise ImportError(
                "Could not import open3d python package. "
                "Please install it with `pip install open3d`."
            )
        import open3d

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

        if not checker.dynamic_import_to_globals(["pysdf"]):
            raise ImportError(
                "Could not import pysdf python package. "
                "Please install open3d with `pip install pysdf`."
            )
        import pysdf

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
            np.ndarray: SDF values of input points without squared, the shape is [N, 1].

        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ImportError(
                "Could not import pymesh python package."
                "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import pymesh

        sdf, _, _, _ = pymesh.signed_distance_to_mesh(self.py_mesh, points)
        sdf = sdf[..., np.newaxis].astype(paddle.get_default_dtype())
        return sdf

    def is_inside(self, x):
        # NOTE: point on boundary is included
        return self.pysdf.contains(x)

    def on_boundary(self, x):
        return np.isclose(self.sdf_func(x), 0.0).flatten()

    def translate(self, translation: np.ndarray, relative: bool = True) -> "Mesh":
        """Translate by given offsets.

        NOTE: This API generate a completely new Mesh object with translated geometry,
        without modifying original Mesh object inplace.

        Args:
            translation (np.ndarray): Translation offsets, numpy array of shape (3,):
                [offset_x, offset_y, offset_z].
            relative (bool, optional): Whether translate relatively. Defaults to True.

        Returns:
            Mesh: Translated Mesh object.

        Examples:
            >>> import ppsci
            >>> import pymesh  # doctest: +SKIP
            >>> import numpy as np
            >>> box = pymesh.generate_box_mesh(np.array([0, 0, 0]), np.array([1, 1, 1]))  # doctest: +SKIP
            >>> mesh = ppsci.geometry.Mesh(box)  # doctest: +SKIP
            >>> print(mesh.vertices)  # doctest: +SKIP
            [[0. 0. 0.]
             [1. 0. 0.]
             [1. 1. 0.]
             [0. 1. 0.]
             [0. 0. 1.]
             [1. 0. 1.]
             [1. 1. 1.]
             [0. 1. 1.]]
            >>> print(mesh.translate((-0.5, 0, 0.5), False).vertices) # the center is moved to the translation vector.  # doctest: +SKIP
            [[-1.  -0.5  0. ]
             [ 0.  -0.5  0. ]
             [ 0.   0.5  0. ]
             [-1.   0.5  0. ]
             [-1.  -0.5  1. ]
             [ 0.  -0.5  1. ]
             [ 0.   0.5  1. ]
             [-1.   0.5  1. ]]
            >>> print(mesh.translate((-0.5, 0, 0.5), True).vertices) # the translation vector is directly added to the geometry coordinates  # doctest: +SKIP
            [[-0.5  0.   0.5]
             [ 0.5  0.   0.5]
             [ 0.5  1.   0.5]
             [-0.5  1.   0.5]
             [-0.5  0.   1.5]
             [ 0.5  0.   1.5]
             [ 0.5  1.   1.5]
             [-0.5  1.   1.5]]
        """
        vertices = np.array(self.vertices, dtype=paddle.get_default_dtype())
        faces = np.array(self.faces)

        if not checker.dynamic_import_to_globals(("open3d", "pymesh")):
            raise ImportError(
                "Could not import open3d and pymesh python package. "
                "Please install open3d with `pip install open3d` and "
                "pymesh as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import open3d  # isort:skip
        import pymesh  # isort:skip

        open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(faces),
        )
        open3d_mesh = open3d_mesh.translate(translation, relative)
        translated_mesh = pymesh.form_mesh(
            np.asarray(open3d_mesh.vertices, dtype=paddle.get_default_dtype()), faces
        )
        # Generate a new Mesh object using class method
        return Mesh.from_pymesh(translated_mesh)

    def scale(
        self, scale: float, center: Tuple[float, float, float] = (0, 0, 0)
    ) -> "Mesh":
        """Scale by given scale coefficient and center coordinate.

        NOTE: This API generate a completely new Mesh object with scaled geometry,
        without modifying original Mesh object inplace.

        Args:
            scale (float): Scale coefficient.
            center (Tuple[float,float,float], optional): Center coordinate, [x, y, z].
                Defaults to (0, 0, 0).

        Returns:
            Mesh: Scaled Mesh object.

        Examples:
            >>> import ppsci
            >>> import pymesh  # doctest: +SKIP
            >>> import numpy as np
            >>> box = pymesh.generate_box_mesh(np.array([0, 0, 0]), np.array([1, 1, 1]))  # doctest: +SKIP
            >>> mesh = ppsci.geometry.Mesh(box)  # doctest: +SKIP
            >>> print(mesh.vertices)  # doctest: +SKIP
            [[0. 0. 0.]
             [1. 0. 0.]
             [1. 1. 0.]
             [0. 1. 0.]
             [0. 0. 1.]
             [1. 0. 1.]
             [1. 1. 1.]
             [0. 1. 1.]]
            >>> mesh = mesh.scale(2, (0.25, 0.5, 0.75))  # doctest: +SKIP
            >>> print(mesh.vertices)  # doctest: +SKIP
            [[-0.25 -0.5  -0.75]
             [ 1.75 -0.5  -0.75]
             [ 1.75  1.5  -0.75]
             [-0.25  1.5  -0.75]
             [-0.25 -0.5   1.25]
             [ 1.75 -0.5   1.25]
             [ 1.75  1.5   1.25]
             [-0.25  1.5   1.25]]
        """
        vertices = np.array(self.vertices, dtype=paddle.get_default_dtype())
        faces = np.array(self.faces, dtype=paddle.get_default_dtype())

        if not checker.dynamic_import_to_globals(("open3d", "pymesh")):
            raise ImportError(
                "Could not import open3d and pymesh python package. "
                "Please install open3d with `pip install open3d` and "
                "pymesh as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import open3d  # isort:skip
        import pymesh  # isort:skip

        open3d_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(faces),
        )
        open3d_mesh = open3d_mesh.scale(scale, center)
        scaled_pymesh = pymesh.form_mesh(
            np.asarray(open3d_mesh.vertices, dtype=paddle.get_default_dtype()), faces
        )
        # Generate a new Mesh object using class method
        return Mesh.from_pymesh(scaled_pymesh)

    def uniform_boundary_points(self, n: int):
        """Compute the equi-spaced points on the boundary."""
        return self.pysdf.sample_surface(n)

    def inflated_random_points(self, n, distance, random="pseudo", criteria=None):
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
        all_areas = []
        for _n, _dist in zip(n, distance):
            inflated_mesh = Mesh(inflation.pymesh_inflation(self.py_mesh, _dist))
            points, areas = inflated_mesh.random_points(_n, random, criteria)
            all_points.append(points)
            all_areas.append(areas)

        all_points = np.concatenate(all_points, axis=0)
        all_areas = np.concatenate(all_areas, axis=0)
        return all_points, all_areas

    def _approximate_area(
        self,
        random: str = "pseudo",
        criteria: Optional[Callable] = None,
        n_appr: int = 10000,
    ) -> float:
        """Approximate area with given `criteria` and `n_appr` points by Monte Carlo
        algorithm.

        Args:
            random (str, optional): Random method. Defaults to "pseudo".
            criteria (Optional[Callable]): Criteria function. Defaults to None.
            n_appr (int): Number of points for approximating area. Defaults to 10000.

        Returns:
            np.ndarray: Approximated areas with shape of [n_faces, ].
        """
        triangle_areas = area_of_triangles(self.v0, self.v1, self.v2)
        triangle_probabilities = triangle_areas / np.linalg.norm(triangle_areas, ord=1)
        triangle_index = np.arange(triangle_probabilities.shape[0])
        npoint_per_triangle = np.random.choice(
            triangle_index, n_appr, p=triangle_probabilities
        )
        npoint_per_triangle, _ = np.histogram(
            npoint_per_triangle,
            np.arange(triangle_probabilities.shape[0] + 1) - 0.5,
        )

        appr_areas = []
        if criteria is not None:
            aux_points = []

        for i, npoint in enumerate(npoint_per_triangle):
            if npoint == 0:
                continue
            # sample points for computing criteria mask if criteria is given
            if criteria is not None:
                points_at_triangle_i = sample_in_triangle(
                    self.v0[i], self.v1[i], self.v2[i], npoint, random
                )
                aux_points.append(points_at_triangle_i)

            appr_areas.append(
                np.full(
                    (npoint, 1), triangle_areas[i] / npoint, paddle.get_default_dtype()
                )
            )
        appr_areas = np.concatenate(appr_areas, axis=0)  # [n_appr, 1]

        # set invalid area to 0 by computing criteria mask with auxiliary points
        if criteria is not None:
            aux_points = np.concatenate(aux_points, axis=0)  # [n_appr, 3]
            criteria_mask = criteria(*np.split(aux_points, self.ndim, 1))
            appr_areas *= criteria_mask
        return appr_areas.sum()

    def random_boundary_points(self, n, random="pseudo"):
        triangle_area = area_of_triangles(self.v0, self.v1, self.v2)
        triangle_prob = triangle_area / np.linalg.norm(triangle_area, ord=1)
        npoint_per_triangle = np.random.choice(
            np.arange(len(triangle_prob)), n, p=triangle_prob
        )
        npoint_per_triangle, _ = np.histogram(
            npoint_per_triangle, np.arange(len(triangle_prob) + 1) - 0.5
        )

        points = []
        normal = []
        areas = []
        for i, npoint in enumerate(npoint_per_triangle):
            if npoint == 0:
                continue
            points_at_triangle_i = sample_in_triangle(
                self.v0[i], self.v1[i], self.v2[i], npoint, random
            )
            normal_at_triangle_i = np.tile(self.face_normal[i], (npoint, 1)).astype(
                paddle.get_default_dtype()
            )
            areas_at_triangle_i = np.full(
                (npoint, 1),
                triangle_area[i] / npoint,
                dtype=paddle.get_default_dtype(),
            )

            points.append(points_at_triangle_i)
            normal.append(normal_at_triangle_i)
            areas.append(areas_at_triangle_i)

        points = np.concatenate(points, axis=0)
        normal = np.concatenate(normal, axis=0)
        areas = np.concatenate(areas, axis=0)

        return points, normal, areas

    def sample_boundary(
        self, n, random="pseudo", criteria=None, evenly=False, inflation_dist=None
    ) -> Dict[str, np.ndarray]:
        # TODO(sensen): support for time-dependent points(repeat data in time)
        if inflation_dist is not None:
            if not isinstance(n, (tuple, list)):
                n = [n]
            if not isinstance(inflation_dist, (tuple, list)):
                inflation_dist = [inflation_dist]
            if len(n) != len(inflation_dist):
                raise ValueError(
                    f"len(n)({len(n)}) should be equal to len(inflation_dist)({len(inflation_dist)})"
                )

            from ppsci.geometry import inflation

            inflated_data_dict = {}
            for _n, _dist in zip(n, inflation_dist):
                # 1. manually inflate mesh at first
                inflated_mesh = Mesh(inflation.pymesh_inflation(self.py_mesh, _dist))
                # 2. compute all data by sample_boundary with `inflation_dist=None`
                data_dict = inflated_mesh.sample_boundary(
                    _n,
                    random,
                    criteria,
                    evenly,
                    inflation_dist=None,
                )
                for key, value in data_dict.items():
                    if key not in inflated_data_dict:
                        inflated_data_dict[key] = value
                    else:
                        inflated_data_dict[key] = np.concatenate(
                            (inflated_data_dict[key], value), axis=0
                        )
            return inflated_data_dict
        else:
            if evenly:
                raise ValueError(
                    "Can't sample evenly on mesh now, please set evenly=False."
                )
            _size, _ntry, _nsuc = 0, 0, 0
            all_points = []
            all_normal = []
            while _size < n:
                points, normal, _ = self.random_boundary_points(n, random)
                if criteria is not None:
                    criteria_mask = criteria(
                        *np.split(points, self.ndim, axis=1)
                    ).flatten()
                    points = points[criteria_mask]
                    normal = normal[criteria_mask]

                if len(points) > n - _size:
                    points = points[: n - _size]
                    normal = normal[: n - _size]

                all_points.append(points)
                all_normal.append(normal)

                _size += len(points)
                _ntry += 1
                if len(points) > 0:
                    _nsuc += 1

                if _ntry >= 1000 and _nsuc == 0:
                    raise ValueError(
                        "Sample boundary points failed, "
                        "please check correctness of geometry and given criteria."
                    )

            all_points = np.concatenate(all_points, axis=0)
            all_normal = np.concatenate(all_normal, axis=0)
            appr_area = self._approximate_area(random, criteria)
            all_areas = np.full((n, 1), appr_area / n, paddle.get_default_dtype())

        x_dict = misc.convert_to_dict(all_points, self.dim_keys)
        normal_dict = misc.convert_to_dict(
            all_normal, [f"normal_{key}" for key in self.dim_keys if key != "t"]
        )
        area_dict = misc.convert_to_dict(all_areas, ["area"])
        return {**x_dict, **normal_dict, **area_dict}

    def random_points(self, n, random="pseudo", criteria=None):
        _size = 0
        all_points = []
        cuboid = geometry_3d.Cuboid(
            [bound[0] for bound in self.bounds],
            [bound[1] for bound in self.bounds],
        )
        _nsample, _nvalid = 0, 0
        while _size < n:
            random_points = cuboid.random_points(n, random)
            valid_mask = self.is_inside(random_points)

            if criteria:
                valid_mask &= criteria(
                    *np.split(random_points, self.ndim, axis=1)
                ).flatten()
            valid_points = random_points[valid_mask]
            _nvalid += len(valid_points)

            if len(valid_points) > n - _size:
                valid_points = valid_points[: n - _size]

            all_points.append(valid_points)
            _size += len(valid_points)
            _nsample += n

        all_points = np.concatenate(all_points, axis=0)
        cuboid_volume = np.prod([b[1] - b[0] for b in self.bounds])
        all_areas = np.full(
            (n, 1), cuboid_volume * (_nvalid / _nsample) / n, paddle.get_default_dtype()
        )
        return all_points, all_areas

    def sample_interior(
        self,
        n,
        random="pseudo",
        criteria=None,
        evenly=False,
        compute_sdf_derivatives: bool = False,
    ):
        """Sample random points in the geometry and return those meet criteria."""
        if evenly:
            # TODO(sensen): implement uniform sample for mesh interior.
            raise NotImplementedError(
                "uniformly sample for interior in mesh is not support yet, "
                "you may need to set evenly=False in config dict of constraint"
            )
        points, areas = self.random_points(n, random, criteria)

        x_dict = misc.convert_to_dict(points, self.dim_keys)
        area_dict = misc.convert_to_dict(areas, ("area",))

        # NOTE: add negative to the sdf values because weight should be positive.
        sdf = -self.sdf_func(points)
        sdf_dict = misc.convert_to_dict(sdf, ("sdf",))

        sdf_derives_dict = {}
        if compute_sdf_derivatives:
            sdf_derives = -self.sdf_derivatives(points)
            sdf_derives_dict = misc.convert_to_dict(
                sdf_derives, tuple(f"sdf__{key}" for key in self.dim_keys)
            )

        return {**x_dict, **area_dict, **sdf_dict, **sdf_derives_dict}

    def union(self, other: "Mesh"):
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ImportError(
                "Could not import pymesh python package. "
                "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import pymesh

        csg = pymesh.CSGTree(
            {"union": [{"mesh": self.py_mesh}, {"mesh": other.py_mesh}]}
        )
        return Mesh(csg.mesh)

    def __or__(self, other: "Mesh"):
        return self.union(other)

    def __add__(self, other: "Mesh"):
        return self.union(other)

    def difference(self, other: "Mesh"):
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ImportError(
                "Could not import pymesh python package. "
                "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import pymesh

        csg = pymesh.CSGTree(
            {"difference": [{"mesh": self.py_mesh}, {"mesh": other.py_mesh}]}
        )
        return Mesh(csg.mesh)

    def __sub__(self, other: "Mesh"):
        return self.difference(other)

    def intersection(self, other: "Mesh"):
        if not checker.dynamic_import_to_globals(["pymesh"]):
            raise ImportError(
                "Could not import pymesh python package. "
                "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
            )
        import pymesh

        csg = pymesh.CSGTree(
            {"intersection": [{"mesh": self.py_mesh}, {"mesh": other.py_mesh}]}
        )
        return Mesh(csg.mesh)

    def __and__(self, other: "Mesh"):
        return self.intersection(other)

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
    """Ref https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle

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
        np.ndarray: Coordinates of sampled n points with shape of [n, 3].
    """
    xs, ys, zs = [], [], []
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

        xs.append(x)
        ys.append(y)
        zs.append(z)
        _size += len(x)

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    zs = np.concatenate(zs, axis=0)

    return np.stack([xs, ys, zs], axis=1)

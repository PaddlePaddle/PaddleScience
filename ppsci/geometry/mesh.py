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

try:
    from stl import mesh as np_mesh_module
except ModuleNotFoundError:
    pass
except ImportError:
    pass

from typing_extensions import Literal

from ppsci.geometry import geometry
from ppsci.geometry import geometry_3d
from ppsci.geometry import sampler
from ppsci.geometry import sdf as sdf_module
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
        return np.isclose(self.sdf_func(x), 0.0).ravel()

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
                "pymesh as https://paddlescience-docs.readthedocs.io/zh/latest/zh/install_setup/#__tabbed_4_1"
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
        random: Literal["pseudo"] = "pseudo",
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
            float: Approximation area with given criteria.
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
        self,
        n: int,
        random: Literal["pseudo"] = "pseudo",
        criteria: Optional[Callable[..., np.ndarray]] = None,
        evenly: bool = False,
        inflation_dist: Union[float, Tuple[float, ...]] = None,
    ) -> Dict[str, np.ndarray]:
        # TODO(sensen): Support for time-dependent points(repeat data in time)
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
                    ).ravel()
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
                ).ravel()
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
        n: int,
        random: Literal["pseudo"] = "pseudo",
        criteria: Optional[Callable[..., np.ndarray]] = None,
        evenly: bool = False,
        compute_sdf_derivatives: bool = False,
    ):
        """Sample random points in the geometry and return those meet criteria."""
        if evenly:
            # TODO(sensen): Implement uniform sample for mesh interior.
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


class SDFMesh(geometry.Geometry):
    """Class for SDF geometry, a kind of implicit surface mesh.

    Args:
        vectors (np.ndarray): Vectors of triangles of mesh with shape [M, 3, 3].
        normals (np.ndarray): Unit normals of each triangle face with shape [M, 3].
        sdf_func (Callable[[np.ndarray, bool], np.ndarray]): Signed distance function
            of the triangle mesh.

    Examples:
        >>> import ppsci
        >>> geom = ppsci.geometry.SDFMesh.from_stl("/path/to/mesh.stl")  # doctest: +SKIP
    """

    eps = 1e-6

    def __init__(
        self,
        vectors: np.ndarray,
        normals: np.ndarray,
        sdf_func: Callable[[np.ndarray, bool], np.ndarray],
    ):
        if vectors.shape[1:] != (3, 3):
            raise ValueError(
                f"The shape of `vectors` must be [M, 3, 3], but got {vectors.shape}"
            )
        if normals.shape[1] != 3:
            raise ValueError(
                f"The shape of `normals` must be [M, 3], but got {normals.shape}"
            )
        self.vectors = vectors
        self.face_normal = normals
        self.sdf_func = sdf_func  # overwrite sdf_func
        self.bounds = (
            ((np.min(self.vectors[:, :, 0])), np.max(self.vectors[:, :, 0])),
            ((np.min(self.vectors[:, :, 1])), np.max(self.vectors[:, :, 1])),
            ((np.min(self.vectors[:, :, 2])), np.max(self.vectors[:, :, 2])),
        )
        self.ndim = 3
        super().__init__(
            self.vectors.shape[-1],
            (np.amin(self.vectors, axis=(0, 1)), np.amax(self.vectors, axis=(0, 1))),
            np.inf,
        )

    @property
    def v0(self) -> np.ndarray:
        return self.vectors[:, 0]

    @property
    def v1(self) -> np.ndarray:
        return self.vectors[:, 1]

    @property
    def v2(self) -> np.ndarray:
        return self.vectors[:, 2]

    @classmethod
    def from_stl(cls, mesh_file: str) -> "SDFMesh":
        """Instantiate SDFMesh from given mesh file.

        Args:
            mesh_file (str): Path to triangle mesh file.

        Returns:
            SDFMesh: Instantiated ppsci.geometry.SDFMesh object.

        Examples:
            >>> import ppsci
            >>> import pymesh  # doctest: +SKIP
            >>> import numpy as np  # doctest: +SKIP
            >>> box = pymesh.generate_box_mesh(np.array([0, 0, 0]), np.array([1, 1, 1]))  # doctest: +SKIP
            >>> pymesh.save_mesh("box.stl", box)  # doctest: +SKIP
            >>> mesh = ppsci.geometry.SDFMesh.from_stl("box.stl")  # doctest: +SKIP
            >>> print(sdfmesh.vectors.shape)  # doctest: +SKIP
            (12, 3, 3)
        """
        # check if pymesh is installed when using Mesh Class
        if not checker.dynamic_import_to_globals(["stl"]):
            raise ImportError(
                "Could not import stl python package. "
                "Please install numpy-stl with: pip install 'numpy-stl>=2.16,<2.17'"
            )

        np_mesh_obj = np_mesh_module.Mesh.from_file(mesh_file)
        return cls(
            np_mesh_obj.vectors,
            np_mesh_obj.get_unit_normals(),
            make_sdf(np_mesh_obj.vectors),
        )

    def sdf_func(
        self, points: np.ndarray, compute_sdf_derivatives: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute signed distance field.

        Args:
            points (np.ndarray): The coordinate points used to calculate the SDF value,
                the shape is [N, 3]
            compute_sdf_derivatives (bool): Whether to compute SDF derivatives.
                Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            If compute_sdf_derivatives is True, then return both SDF values([N, 1])
                and their derivatives([N, 3]); otherwise only return SDF values([N, 1]).

        NOTE: This function usually returns ndarray with negative values, because
        according to the definition of SDF, the SDF value of the coordinate point inside
        the object(interior points) is negative, the outside is positive, and the edge
        is 0. Therefore, when used for weighting, a negative sign is often added before
        the result of this function.
        """
        # normalize triangles
        x_min, y_min, z_min = np.min(points, axis=0)
        x_max, y_max, z_max = np.max(points, axis=0)
        max_dis = max(max((x_max - x_min), (y_max - y_min)), (z_max - z_min))
        store_triangles = np.array(self.vectors, dtype=np.float64)
        store_triangles[:, :, 0] -= x_min
        store_triangles[:, :, 1] -= y_min
        store_triangles[:, :, 2] -= z_min
        store_triangles *= 1 / max_dis
        store_triangles = store_triangles.reshape([-1, 3])

        # normalize query points
        points = points.copy()
        points[:, 0] -= x_min
        points[:, 1] -= y_min
        points[:, 2] -= z_min
        points *= 1 / max_dis
        points = points.astype(np.float64).ravel()

        # compute sdf values for query points
        sdf = sdf_module.signed_distance_field(
            store_triangles,
            np.arange((store_triangles.shape[0])),
            points,
            include_hit_points=compute_sdf_derivatives,
        )
        if compute_sdf_derivatives:
            sdf, hit_points = sdf

        sdf = sdf.numpy()  # [N]
        sdf = np.expand_dims(max_dis * sdf, axis=1)  # [N, 1]

        if compute_sdf_derivatives:
            hit_points = hit_points.numpy()  # [N, 3]
            # Gradient of SDF is the unit vector from the query point to the hit point.
            sdf_derives = hit_points - points
            sdf_derives /= np.linalg.norm(sdf_derives, axis=1, keepdims=True)
            return sdf, sdf_derives

        return sdf

    def is_inside(self, x):
        # NOTE: point on boundary is included
        return np.less(self.sdf_func(x), 0.0).ravel()

    def on_boundary(self, x: np.ndarray, normal: np.ndarray) -> np.ndarray:
        x_plus = x + self.eps * normal
        x_minus = x - self.eps * normal

        sdf_x_plus = self.sdf_func(x_plus)
        sdf_x_minus = self.sdf_func(x_minus)
        mask_on_boundary = np.less_equal(sdf_x_plus * sdf_x_minus, 0)
        return mask_on_boundary.ravel()

    def translate(self, translation: np.ndarray) -> "SDFMesh":
        """Translate by given offsets.

        NOTE: This API generate a completely new Mesh object with translated geometry,
        without modifying original Mesh object inplace.

        Args:
            translation (np.ndarray): Translation offsets, numpy array of shape (3,):
                [offset_x, offset_y, offset_z].

        Returns:
            Mesh: Translated Mesh object.

        Examples:
            >>> import ppsci
            >>> import pymesh  # doctest: +SKIP
            >>> mesh = ppsci.geometry.SDFMesh.from_stl('/path/to/mesh.stl')  # doctest: +SKIP
            >>> mesh = mesh.translate(np.array([1, -1, 2]))  # doctest: +SKIP
        """
        new_vectors = self.vectors + translation.reshape([1, 1, 3])

        return SDFMesh(
            new_vectors,
            self.face_normal,
            make_sdf(new_vectors),
        )

    def scale(self, scale: float) -> "SDFMesh":
        """Scale by given scale coefficient and center coordinate.

        NOTE: This API generate a completely new Mesh object with scaled geometry,
        without modifying original Mesh object inplace.

        Args:
            scale (float): Scale coefficient.

        Returns:
            Mesh: Scaled Mesh object.

        Examples:
            >>> import ppsci
            >>> import pymesh  # doctest: +SKIP
            >>> mesh = ppsci.geometry.SDFMesh.from_stl('/path/to/mesh.stl')  # doctest: +SKIP
            >>> mesh = mesh.scale(np.array([1.3, 1.5, 2.0]))  # doctest: +SKIP
        """
        new_vectors = self.vectors * scale
        return SDFMesh(
            new_vectors,
            self.face_normal,
            make_sdf(new_vectors),
        )

    def uniform_boundary_points(self, n: int):
        """Compute the equi-spaced points on the boundary."""
        raise NotImplementedError(
            "'uniform_boundary_points' is not available in SDFMesh."
        )

    def inflated_random_points(self, n, distance, random="pseudo", criteria=None):
        raise NotImplementedError(
            "'inflated_random_points' is not available in SDFMesh."
        )

    def _approximate_area(
        self,
        random: Literal["pseudo"] = "pseudo",
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
            float: Approximation area with given criteria.
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

        aux_points = []
        aux_normals = []
        appr_areas = []

        for i, npoint in enumerate(npoint_per_triangle):
            if npoint == 0:
                continue
            # sample points for computing criteria mask if criteria is given
            points_at_triangle_i = sample_in_triangle(
                self.v0[i], self.v1[i], self.v2[i], npoint, random
            )
            normal_at_triangle_i = np.tile(
                self.face_normal[i].reshape(1, 3), (npoint, 1)
            )
            aux_points.append(points_at_triangle_i)
            aux_normals.append(normal_at_triangle_i)
            appr_areas.append(
                np.full(
                    (npoint, 1), triangle_areas[i] / npoint, paddle.get_default_dtype()
                )
            )

        aux_points = np.concatenate(aux_points, axis=0)  # [n_appr, 3]
        aux_normals = np.concatenate(aux_normals, axis=0)  # [n_appr, 3]
        appr_areas = np.concatenate(appr_areas, axis=0)  # [n_appr, 1]
        valid_mask = self.on_boundary(aux_points, aux_normals)[:, None]
        # set invalid area to 0 by computing criteria mask with auxiliary points
        if criteria is not None:
            criteria_mask = criteria(*np.split(aux_points, self.ndim, 1))
            assert valid_mask.shape == criteria_mask.shape
            valid_mask = np.logical_and(valid_mask, criteria_mask)

        appr_areas *= valid_mask

        return appr_areas.sum()

    def random_boundary_points(
        self, n, random="pseudo"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        self,
        n: int,
        random: Literal["pseudo"] = "pseudo",
        criteria: Optional[Callable[..., np.ndarray]] = None,
        evenly: bool = False,
        inflation_dist: Union[float, Tuple[float, ...]] = None,
    ) -> Dict[str, np.ndarray]:
        # TODO(sensen): Support for time-dependent points(repeat data in time)
        if inflation_dist is not None:
            raise NotImplementedError("Not implemented yet")
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
                valid_mask = self.on_boundary(points, normal)

                if criteria is not None:
                    criteria_mask = criteria(
                        *np.split(points, self.ndim, axis=1)
                    ).ravel()
                    assert valid_mask.shape == criteria_mask.shape
                    valid_mask = np.logical_and(valid_mask, criteria_mask)

                points = points[valid_mask]
                normal = normal[valid_mask]

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
            _appr_area = self._approximate_area(random, criteria)
            all_areas = np.full((n, 1), _appr_area / n, paddle.get_default_dtype())

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
                criteria_mask = criteria(
                    *np.split(random_points, self.ndim, axis=1)
                ).ravel()
                assert valid_mask.shape == criteria_mask.shape
                valid_mask = np.logical_and(valid_mask, criteria_mask)

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
        n: int,
        random: Literal["pseudo"] = "pseudo",
        criteria: Optional[Callable[..., np.ndarray]] = None,
        evenly: bool = False,
        compute_sdf_derivatives: bool = False,
    ):
        """Sample random points in the geometry and return those meet criteria."""
        if evenly:
            # TODO(sensen): Implement uniform sample for mesh interior.
            raise NotImplementedError(
                "uniformly sample for interior in mesh is not support yet, "
                "you may need to set evenly=False in config dict of constraint"
            )
        points, areas = self.random_points(n, random, criteria)

        x_dict = misc.convert_to_dict(points, self.dim_keys)
        area_dict = misc.convert_to_dict(areas, ("area",))

        sdf = self.sdf_func(points, compute_sdf_derivatives)
        if compute_sdf_derivatives:
            sdf, sdf_derives = sdf

        # NOTE: Negate sdf because weight should be positive.
        sdf_dict = misc.convert_to_dict(-sdf, ("sdf",))

        sdf_derives_dict = {}
        if compute_sdf_derivatives:
            # NOTE: Negate sdf derivatives
            sdf_derives_dict = misc.convert_to_dict(
                -sdf_derives, tuple(f"sdf__{key}" for key in self.dim_keys)
            )

        return {**x_dict, **area_dict, **sdf_dict, **sdf_derives_dict}

    def union(self, other: "SDFMesh"):
        new_vectors = np.concatenate([self.vectors, other.vectors], axis=0)
        new_normals = np.concatenate([self.face_normal, other.face_normal], axis=0)

        def make_union_new_sdf(sdf_func1, sdf_func2):
            def new_sdf_func(points: np.ndarray, compute_sdf_derivatives: bool = False):
                # Invert definition of sdf to make boolean operation accurate
                # see: https://iquilezles.org/articles/interiordistance/
                sdf_self = sdf_func1(points, compute_sdf_derivatives)
                sdf_other = sdf_func2(points, compute_sdf_derivatives)
                if compute_sdf_derivatives:
                    sdf_self, sdf_derives_self = sdf_self
                    sdf_other, sdf_derives_other = sdf_other

                computed_sdf = -np.maximum(-sdf_self, -sdf_other)

                if compute_sdf_derivatives:
                    computed_sdf_derives = -np.where(
                        sdf_self < sdf_other,
                        sdf_derives_self,
                        sdf_derives_other,
                    )
                    return computed_sdf, computed_sdf_derives

                return computed_sdf

            return new_sdf_func

        return SDFMesh(
            new_vectors,
            new_normals,
            make_union_new_sdf(self.sdf_func, other.sdf_func),
        )

    def __or__(self, other: "SDFMesh"):
        return self.union(other)

    def __add__(self, other: "SDFMesh"):
        return self.union(other)

    def difference(self, other: "SDFMesh"):
        new_vectors = np.concatenate([self.vectors, other.vectors], axis=0)
        new_normals = np.concatenate([self.face_normal, -other.face_normal], axis=0)

        def make_difference_new_sdf(sdf_func1, sdf_func2):
            def new_sdf_func(points: np.ndarray, compute_sdf_derivatives: bool = False):
                # Invert definition of sdf to make boolean operation accurate
                # see: https://iquilezles.org/articles/interiordistance/
                sdf_self = sdf_func1(points, compute_sdf_derivatives)
                sdf_other = sdf_func2(points, compute_sdf_derivatives)
                if compute_sdf_derivatives:
                    sdf_self, sdf_derives_self = sdf_self
                    sdf_other, sdf_derives_other = sdf_other

                computed_sdf = -np.minimum(-sdf_self, sdf_other)

                if compute_sdf_derivatives:
                    computed_sdf_derives = np.where(
                        -sdf_self < sdf_other,
                        -sdf_derives_self,
                        sdf_derives_other,
                    )
                    return computed_sdf, computed_sdf_derives

                return computed_sdf

            return new_sdf_func

        return SDFMesh(
            new_vectors,
            new_normals,
            make_difference_new_sdf(self.sdf_func, other.sdf_func),
        )

    def __sub__(self, other: "SDFMesh"):
        return self.difference(other)

    def intersection(self, other: "SDFMesh"):
        new_vectors = np.concatenate([self.vectors, other.vectors], axis=0)
        new_normals = np.concatenate([self.face_normal, other.face_normal], axis=0)

        def make_intersection_new_sdf(sdf_func1, sdf_func2):
            def new_sdf_func(points: np.ndarray, compute_sdf_derivatives: bool = False):
                # Invert definition of sdf to make boolean operation accurate
                # see: https://iquilezles.org/articles/interiordistance/
                sdf_self = sdf_func1(points, compute_sdf_derivatives)
                sdf_other = sdf_func2(points, compute_sdf_derivatives)
                if compute_sdf_derivatives:
                    sdf_self, sdf_derives_self = sdf_self
                    sdf_other, sdf_derives_other = sdf_other

                computed_sdf = -np.minimum(-sdf_self, -sdf_other)

                if compute_sdf_derivatives:
                    computed_sdf_derives = np.where(
                        sdf_self > sdf_other,
                        -sdf_derives_self,
                        -sdf_derives_other,
                    )
                    return computed_sdf, computed_sdf_derives

                return computed_sdf

            return new_sdf_func

        return SDFMesh(
            new_vectors,
            new_normals,
            make_intersection_new_sdf(self.sdf_func, other.sdf_func),
        )

    def __and__(self, other: "SDFMesh"):
        return self.intersection(other)

    def __str__(self) -> str:
        """Return the name of class"""
        return ", ".join(
            [
                self.__class__.__name__,
                f"num_faces = {self.vectors.shape[0]}",
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
        r1 = sampler.sample(n, 1, random).ravel()
        r2 = sampler.sample(n, 1, random).ravel()
        s1 = np.sqrt(r1)
        x = v0[0] * (1.0 - s1) + v1[0] * (1.0 - r2) * s1 + v2[0] * r2 * s1
        y = v0[1] * (1.0 - s1) + v1[1] * (1.0 - r2) * s1 + v2[1] * r2 * s1
        z = v0[2] * (1.0 - s1) + v1[2] * (1.0 - r2) * s1 + v2[2] * r2 * s1

        if criteria is not None:
            criteria_mask = criteria(x, y, z).ravel()
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


def make_sdf(vectors: np.ndarray):
    def sdf_func(points: np.ndarray, compute_sdf_derivatives=False):
        points = points.copy()
        x_min, y_min, z_min = np.min(points, axis=0)
        x_max, y_max, z_max = np.max(points, axis=0)
        max_dis = max(max((x_max - x_min), (y_max - y_min)), (z_max - z_min))
        store_triangles = vectors.copy()
        store_triangles[:, :, 0] -= x_min
        store_triangles[:, :, 1] -= y_min
        store_triangles[:, :, 2] -= z_min
        store_triangles *= 1 / max_dis
        store_triangles = store_triangles.reshape([-1, 3])
        points[:, 0] -= x_min
        points[:, 1] -= y_min
        points[:, 2] -= z_min
        points *= 1 / max_dis
        points = points.astype(np.float64).ravel()

        # compute sdf values
        sdf = sdf_module.signed_distance_field(
            store_triangles,
            np.arange((store_triangles.shape[0])),
            points,
            include_hit_points=compute_sdf_derivatives,
        )
        if compute_sdf_derivatives:
            sdf, sdf_derives = sdf

        sdf = sdf.numpy()
        sdf = np.expand_dims(max_dis * sdf, axis=1)

        if compute_sdf_derivatives:
            sdf_derives = sdf_derives.numpy().reshape(-1)
            sdf_derives = -(sdf_derives - points)
            sdf_derives = np.reshape(sdf_derives, (sdf_derives.shape[0] // 3, 3))
            sdf_derives = sdf_derives / np.linalg.norm(
                sdf_derives, axis=1, keepdims=True
            )
            return sdf, sdf_derives

        return sdf

    return sdf_func

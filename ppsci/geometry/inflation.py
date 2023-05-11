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

import numpy as np
import paddle

from ppsci.utils import checker

if not checker.dynamic_import_to_globals(["pymesh", "open3d"]):
    raise ModuleNotFoundError

__all__ = [
    "pymesh_inflation",
]


def open3d_inflation(
    mesh: open3d.geometry.TriangleMesh, distance: float, direction: int = 1
) -> open3d.geometry.TriangleMesh:
    """Inflate mesh geometry.

    Args:
        mesh (open3d.geometry.TriangleMesh): Open3D mesh object.
        distance (float): Distance along exterior normal to inflate.
        direction (int): 1 for exterior normal, -1 for interior normal. Defaults to 1.

    Returns:
        open3d.geometry.TriangleMesh: Inflated mesh.
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles)
    points = np.asarray(mesh.vertices)

    remove_ids = []
    for i, point in enumerate(points):
        boolean_index = np.argwhere(triangles == i)[:, 0]
        if len(boolean_index) < 3:
            remove_ids.append(i)
    mesh.remove_vertices_by_index(remove_ids)

    points = np.asarray(mesh.vertices, dtype=paddle.get_default_dtype())
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals, dtype=paddle.get_default_dtype())
    mesh.orient_triangles()
    triangles = np.asarray(mesh.triangles, dtype=paddle.get_default_dtype())
    new_points = []
    for i, point in enumerate(points):
        boolean_index = np.argwhere(triangles == i)[:, 0]
        normal = normals[boolean_index] * direction
        d = np.ones(len(normal), dtype=paddle.get_default_dtype()) * distance

        new_point = np.linalg.lstsq(normal, d, rcond=None)[0].squeeze()
        new_point = point + new_point
        if np.linalg.norm(new_point - point) > distance * 2:
            # TODO : Find a better way to solve the bad inflation
            new_point = point + distance * normal.mean(axis=0)

        new_points.append(new_point)

    new_points = np.array(new_points, dtype=paddle.get_default_dtype())
    new_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(new_points),
        open3d.utility.Vector3iVector(triangles),
    )

    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_triangle_normals()
    return new_mesh


def pymesh_inflation(mesh: pymesh.Mesh, distance: float) -> pymesh.Mesh:
    """Inflate mesh by distance.

    Args:
        mesh (pymesh.Mesh): PyMesh object.
        distance (float): Inflation distance.

    Returns:
        pymesh.Mesh: Inflated mesh.
    """
    vertices = np.array(mesh.vertices, dtype=paddle.get_default_dtype())
    faces = np.array(mesh.faces)
    open3d_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(vertices), open3d.utility.Vector3iVector(faces)
    )
    inflated_open3d_mesh = open3d_inflation(
        open3d_mesh, abs(distance), 1.0 if distance >= 0.0 else -1.0
    )
    vertices = np.array(inflated_open3d_mesh.vertices, dtype=paddle.get_default_dtype())
    faces = np.array(inflated_open3d_mesh.triangles)
    inflated_pymesh = pymesh.form_mesh(vertices, faces)
    return inflated_pymesh


def offset(mesh, distance) -> open3d.geometry.TriangleMesh:
    """Offset the 2D mesh

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to be offset.
        distance (float): The distance to offset.

    Returns:
        open3d.geometry.TriangleMesh: Result mesh.
    """
    # check if the mesh is 2D
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals, dtype=paddle.get_default_dtype())
    if not np.allclose(normals[:, :-1], 0):
        raise ValueError("The mesh is not 2D")

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles, dtype=paddle.get_default_dtype())

    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = set(map(tuple, edges))
    edges = np.array(list(edges))

    vertices = np.asarray(mesh.vertices, dtype=paddle.get_default_dtype())[:, :-1]
    edges_in_triangle = np.array(
        [
            np.intersect1d(
                np.argwhere(triangles == edge[0])[:, 0],
                np.argwhere(triangles == edge[1])[:, 0],
            )
            for edge in edges
        ],
        dtype=object,
    )
    surface_edges = edges[[len(i) == 1 for i in edges_in_triangle]]
    edges_in_triangle = [i for i in edges_in_triangle if len(i) == 1]

    edges_normals = []
    for edge, triangle in zip(surface_edges, edges_in_triangle):
        triangle = triangles[triangle].squeeze()
        other_point = vertices[np.setdiff1d(triangle, edge)].squeeze()
        edge = vertices[edge]
        u = (other_point[0] - edge[0][0]) * (edge[0][0] - edge[1][0]) + (
            other_point[1] - edge[0][1]
        ) * (edge[0][1] - edge[1][1])
        u = u / np.sum((edge[0] - edge[1]) ** 2)
        edge_normal = edge[0] + u * (edge[0] - edge[1])
        edge_normal = edge_normal - other_point
        edges_normals.append(edge_normal)

    edges_normals = np.array(edges_normals, dtype=paddle.get_default_dtype())
    edges_normals = edges_normals / np.linalg.norm(edges_normals, axis=1)[:, None]

    new_mesh = open3d.geometry.TriangleMesh()
    new_vertices = []
    for point in set(surface_edges.reshape(-1)):
        index = np.argwhere(surface_edges == point)[:, 0]
        normal = edges_normals[index]
        d = np.ones(len(index), dtype=paddle.get_default_dtype()) * distance
        new_point = np.linalg.lstsq(normal, d, rcond=None)[0]
        new_point = vertices[point] + new_point
        new_vertices.append(new_point)

    new_vertices = np.hstack(
        (
            np.array(new_vertices, dtype=paddle.get_default_dtype()),
            np.zeros((len(new_vertices), 1), dtype=paddle.get_default_dtype()),
        )
    )
    new_mesh.vertices = open3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = open3d.utility.Vector3iVector(triangles)
    new_mesh.compute_triangle_normals()
    new_mesh.compute_vertex_normals()
    return new_mesh

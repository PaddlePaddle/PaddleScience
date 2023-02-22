# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import warnings

import numpy as np

try:
    import open3d
    import pymesh
except ImportError:
    warnings.warn(
        f"Refer to README.md and install open3d and pymesh before using inflation API in neo_geometry"
    )


def inflation(mesh, dis, direction=1):
    """Inflation the mesh

    Args:
        mesh (open3d.geometry.TriangleMesh): Mesh to be inflated.
        dis (float): Distance to inflate.
        direction (int, optional): 1 for outer normal, -1 for inner normal. Defaults to 1.

    Returns:
        open3d.geometry.TriangleMesh: The inflated mesh.
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

    points = np.asarray(mesh.vertices)
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    mesh.orient_triangles()
    triangles = np.asarray(mesh.triangles)
    new_points = []
    for i, point in enumerate(points):
        boolean_index = np.argwhere(triangles == i)[:, 0]
        normal = normals[boolean_index] * direction
        d = np.ones(len(normal)) * dis

        new_point = np.linalg.lstsq(normal, d, rcond=None)[0].squeeze()
        new_point = point + new_point
        if np.linalg.norm(new_point - point) > dis * 2:
            # TODO : Find a better way to solve the bad inflation
            new_point = point + dis * normal.mean(axis=0)

        new_points.append(new_point)

    new_points = np.array(new_points)
    new_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(new_points),
        open3d.utility.Vector3iVector(triangles))

    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_triangle_normals()
    return new_mesh


def pymesh_inflation(mesh, distance):
    """Inflate mesh by distance

    Args:
        mesh (pymesh.Mesh): PyMesh object.
        distance (float): Inflation distance.

    Returns:
        pymesh.Mesh: Inflated mesh.
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    open3d_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(vertices),
        open3d.utility.Vector3iVector(faces))
    inflated_open3d_mesh = inflation(open3d_mesh,
                                     abs(distance), 1.0
                                     if distance >= 0.0 else -1.0)
    vertices = np.array(inflated_open3d_mesh.vertices).astype("float32")
    faces = np.array(inflated_open3d_mesh.triangles)
    inflated_pymesh = pymesh.form_mesh(vertices, faces)
    return inflated_pymesh


def offset(mesh, dis):
    """ Offset the 2D mesh

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to be offset.
        dis (float): The distance to offset.

    Returns:
        open3d.geometry.TriangleMesh: Result mesh.
    """
    # check if the mesh is 2D
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    if not np.allclose(normals[:, :-1], 0):
        raise ValueError("The mesh is not 2D")

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles)

    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = set(map(tuple, edges))
    edges = np.array(list(edges))

    vertices = np.asarray(mesh.vertices)[:, :-1]
    edges_in_triangle = np.array(
        [
            np.intersect1d(
                np.argwhere(triangles == edge[0])[:, 0],
                np.argwhere(triangles == edge[1])[:, 0], ) for edge in edges
        ],
        dtype=object, )
    surface_edges = edges[[len(i) == 1 for i in edges_in_triangle]]
    edges_in_triangle = [i for i in edges_in_triangle if len(i) == 1]

    edges_normals = []
    for edge, triangle in zip(surface_edges, edges_in_triangle):
        triangle = triangles[triangle].squeeze()
        other_point = vertices[np.setdiff1d(triangle, edge)].squeeze()
        edge = vertices[edge]
        u = (other_point[0] - edge[0][0]) * (edge[0][0] - edge[1][0]) + (
            other_point[1] - edge[0][1]) * (edge[0][1] - edge[1][1])
        u = u / np.sum((edge[0] - edge[1])**2)
        edge_normal = edge[0] + u * (edge[0] - edge[1])
        edge_normal = edge_normal - other_point
        edges_normals.append(edge_normal)

    edges_normals = np.array(edges_normals)
    edges_normals = edges_normals / np.linalg.norm(
        edges_normals, axis=1)[:, None]

    new_mesh = open3d.geometry.TriangleMesh()
    new_vertices = []
    for point in set(surface_edges.reshape(-1)):
        index = np.argwhere(surface_edges == point)[:, 0]
        normal = edges_normals[index]
        d = np.ones(len(index)) * dis
        new_point = np.linalg.lstsq(normal, d, rcond=None)[0]
        new_point = vertices[point] + new_point
        new_vertices.append(new_point)

    new_vertices = np.hstack((np.array(new_vertices), np.zeros(
        (len(new_vertices), 1))))
    new_mesh.vertices = open3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = open3d.utility.Vector3iVector(triangles)
    new_mesh.compute_triangle_normals()
    new_mesh.compute_vertex_normals()
    return new_mesh

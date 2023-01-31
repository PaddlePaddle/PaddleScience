import numpy as np
import open3d
import pymesh

from typing import Union


def inflation(mesh, dis, direction=1):
    """
    Inflation the mesh

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to be inflated
    dis : float
        The distance to inflate
    dir: direction of normals, default to 1(outer normal), -1 for inner normal

    Returns
    -------
    open3d.geometry.TriangleMesh
        The inflated mesh
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
        d = np.ones(len(normal)) * dis # [dis, dis, ... dis]
        # 求一个点x，它与所有的法向量点积的结果都是dis
        new_point = np.linalg.lstsq(normal, d, rcond=None)[0].squeeze()
        new_point = point + new_point
        if np.linalg.norm(new_point - point) > dis * 2:
            # TODO : Find a better way to solve the bad inflation
            new_point = point + dis * normal.mean(axis=0)

        new_points.append(new_point)

    new_points = np.array(new_points)
    new_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(new_points),
        open3d.utility.Vector3iVector(triangles)
    )
    # new_mesh.vertices = open3d.utility.Vector3dVector(new_points)
    # new_mesh.triangles = open3d.utility.Vector3iVector(triangles)

    new_mesh.remove_duplicated_vertices()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_triangle_normals()
    return new_mesh


def pymesh_inflation(mesh: pymesh.Mesh, distance: Union[int, float]) -> pymesh.Mesh:
    """inflate mesh by distance

    Args:
        mesh (pymesh.Mesh): PyMesh object.
        distance (int or float): Distance factor for inflation.

    Returns:
        pymesh.Mesh: Inflated mesh.
    """
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    open3d_mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(vertices),
        open3d.utility.Vector3iVector(faces)
    )
    inflated_open3d_mesh = inflation(open3d_mesh, abs(distance), 1.0 if distance >= 0.0 else -1.0)
    vertices = np.array(inflated_open3d_mesh.vertices).astype("float32")
    faces = np.array(inflated_open3d_mesh.triangles)
    inflated_pymesh = pymesh.form_mesh(vertices, faces)
    return inflated_pymesh


def offset(mesh, dis):
    """
    Offset the 2D mesh

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to be offset

    dis : float
        The distance to offset

    Returns
    -------
    open3d.geometry.TriangleMesh
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
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = set(map(tuple, edges))
    edges = np.array(list(edges))

    vertices = np.asarray(mesh.vertices)[:, :-1]
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

    edges_normals = np.array(edges_normals)
    edges_normals = edges_normals / np.linalg.norm(edges_normals, axis=1)[:, None]

    new_mesh = open3d.geometry.TriangleMesh()
    new_vertices = []
    for point in set(surface_edges.reshape(-1)):
        index = np.argwhere(surface_edges == point)[:, 0]
        normal = edges_normals[index]
        d = np.ones(len(index)) * dis
        new_point = np.linalg.lstsq(normal, d, rcond=None)[0]
        new_point = vertices[point] + new_point
        new_vertices.append(new_point)

    new_vertices = np.hstack((np.array(new_vertices), np.zeros((len(new_vertices), 1))))
    new_mesh.vertices = open3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = open3d.utility.Vector3iVector(triangles)
    new_mesh.compute_triangle_normals()
    new_mesh.compute_vertex_normals()
    return new_mesh


def sample_points(mesh, nr_points=1, mode="uniform", seed=-1):
    """
    Sample points from the mesh

    Parameters
    ----------

    mesh : open3d.geometry.TriangleMesh
        The mesh to be sampled

    nr_points : int
        The number of points to be sampled

    mode : str
        The mode of sampling, can be "uniform" or "poisson_disk"

    seed : int
        The seed of random number generator

    Returns
    -------
    open3d.geometry.PointCloud
    """

    if mode == "uniform":
        pc = mesh.sample_points_uniformly(nr_points, seed=seed)
    elif mode == "poisson_disk":
        pc = mesh.sample_points_poisson_disk(nr_points, seed=seed)
    else:
        pass
    return pc


def edge_sample(mesh, nr_points=1, mode="uniform", seed=-1):
    """
    Sample points from the 2D mesh edges

    Parameters
    ----------
    mehs : open3d.geometry.TriangleMesh
        The mesh to be sampled

    nr_points : int
        The number of points to be sampled

    mode : str
        The mode of sampling, can be "uniform" or "grid"

    seed : int
        The seed of random number generator
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles)

    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = set(map(tuple, edges))
    edges = np.array(list(edges))

    vertices = np.asarray(mesh.vertices)
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
    edges_points = vertices[surface_edges]
    vlist = []
    for edge in edges_points:
        dx = edge[1][0] - edge[0][0]
        dy = edge[1][1] - edge[0][1]
        lenght = np.sqrt(dx**2 + dy**2)
        if np.isclose(dx, 0):
            y = sample(1, nr_points, edge[0][1], edge[1][1], type=mode)
            x = np.full_like(y, edge[0][0])
        elif np.isclose(dy, 0):
            x = sample(1, nr_points, edge[0][0], edge[1][0], type=mode)
            y = np.full_like(x, edge[0][1])
        else:
            k = dy / dx
            b = edge[0][1] - k * edge[0][0]
            x = sample(1, nr_points, edge[0][0], edge[1][0], type=mode)
            y = k * x + b
        vlist.append(np.hstack([x, y, np.zeros_like(x)]))

    vlist = np.vstack(vlist)
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(vlist)
    return pc


def sample(ndim, nr_points, l_bound, u_bound, type="uniform"):
    if l_bound > u_bound:
        l_bound, u_bound = u_bound, l_bound
    if type == "uniform":
        return np.random.uniform(l_bound, u_bound, (nr_points, ndim))
    elif type == "grid":
        return np.linspace(l_bound, u_bound, nr_points).reshape(-1, 1)
    else:
        raise ValueError("type must be uniform or grid")


def inflation_sample(
    mesh,
    dist_list=(float, list, np.ndarray),
    nr_points_list=(float, list, np.ndarray),
    dim=3,
    mode="uniform",
    seed=-1,
):
    """
    Inflation the mesh and sample points from the inflated mesh

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh

    dist_list : float or list or np.ndarray
        The distance to inflate

    nr_points_list : float or list or np.ndarray
        The list of numbers for points to be sampled

    dim : int
        The dimension of the mesh, can be 2 or 3

    mode : str
        The mode of sampling, can be "uniform" or "poisson_disk"

    seed : int
        The seed of random number generator

    Returns
    -------
    open3d.geometry.PointCloud
    """
    assert dim in [2, 3], "The dimension of the mesh can only be 2 or 3"
    if isinstance(dist_list, (float, int)):
        dist_list = [dist_list]
    if isinstance(nr_points_list, (float, int)):
        nr_points_list = [nr_points_list]
    if len(dist_list) != len(nr_points_list):
        raise ValueError("The length of dist_list and nr_points_list must be equal")
    new_mesh = mesh
    pcs = []
    for dis, nr_points in zip(dist_list, nr_points_list):
        if dim == 2:
            new_mesh = offset(new_mesh, dis)
            pc = edge_sample(new_mesh, nr_points, mode, seed)
        else:
            new_mesh = inflation(new_mesh, dis)
            pc = sample_points(new_mesh, nr_points, mode, seed)
        pc.paint_uniform_color(np.random.random(3))
        pcs.append(pc)
    return pcs

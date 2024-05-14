# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import pickle
import typing

import args
import numpy as np
import paddle
import pandas as pd
import xarray
from paddle import io

import itertools
import typing

import numpy as np
import paddle
import scipy
import trimesh
import utils
import xarray


class GraphGridMesh(object):
    def __init__(
        self,
        config,
        mesh2mesh_src_index=None,
        mesh2mesh_dst_index=None,
        grid2mesh_src_index=None,
        grid2mesh_dst_index=None,
        mesh2grid_src_index=None,
        mesh2grid_dst_index=None,
        mesh_num_nodes=None,
        grid_num_nodes=None,
        mesh_num_edges=None,
        grid2mesh_num_edges=None,
        mesh2grid_num_edges=None,
        grid_node_feat=None,
        mesh_node_feat=None,
        mesh_edge_feat=None,
        grid2mesh_edge_feat=None,
        mesh2grid_edge_feat=None,
    ):
        """_summary_

        Args:
            config (_type_): _description_
            mesh2mesh_src_index (_type_, optional): _description_. Defaults to None.
            mesh2mesh_dst_index (_type_, optional): _description_. Defaults to None.
            grid2mesh_src_index (_type_, optional): _description_. Defaults to None.
            grid2mesh_dst_index (_type_, optional): _description_. Defaults to None.
            mesh2grid_src_index (_type_, optional): _description_. Defaults to None.
            mesh2grid_dst_index (_type_, optional): _description_. Defaults to None.
            mesh_num_nodes (_type_, optional): _description_. Defaults to None.
            grid_num_nodes (_type_, optional): _description_. Defaults to None.
            mesh_num_edges (_type_, optional): _description_. Defaults to None.
            grid2mesh_num_edges (_type_, optional): _description_. Defaults to None.
            mesh2grid_num_edges (_type_, optional): _description_. Defaults to None.
            grid_node_feat (_type_, optional): _description_. Defaults to None.
            mesh_node_feat (_type_, optional): _description_. Defaults to None.
            mesh_edge_feat (_type_, optional): _description_. Defaults to None.
            grid2mesh_edge_feat (_type_, optional): _description_. Defaults to None.
            mesh2grid_edge_feat (_type_, optional): _description_. Defaults to None.
        """
        self.meshes = get_hierarchy_of_triangular_meshes_for_sphere(config.mesh_size)

        all_input_vars = [
            mesh2mesh_src_index,
            mesh2mesh_dst_index,
            grid2mesh_src_index,
            grid2mesh_dst_index,
            mesh2grid_src_index,
            mesh2grid_dst_index,
            mesh_num_nodes,
            grid_num_nodes,
            mesh_num_edges,
            grid2mesh_num_edges,
            mesh2grid_num_edges,
            grid_node_feat,
            mesh_node_feat,
            mesh_edge_feat,
            grid2mesh_edge_feat,
            mesh2grid_edge_feat,
        ]
        should_init = any(var is None for var in all_input_vars)

        if should_init:
            # 初始化构建
            self.query_radius = (
                self._get_max_edge_distance(self.finest_mesh)
                * config.radius_query_fraction_edge_length
            )
            self._mesh2grid_edge_normalization_factor = (
                config.mesh2grid_edge_normalization_factor
            )
            self._spatial_features_kwargs = dict(
                add_node_positions=False,
                add_node_latitude=True,
                add_node_longitude=True,
                add_relative_positions=True,
                relative_longitude_local_coordinates=True,
                relative_latitude_local_coordinates=True,
            )

            self.init_mesh_properties()
            self._init_grid_properties(
                grid_lat=np.arange(-90.0, 90.0 + config.resolution, config.resolution),
                grid_lon=np.arange(0.0, 360.0, config.resolution),
            )
            self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
            self._mesh_graph_structure = self._init_mesh_graph()
            self._mesh2grid_graph_structure = self._init_mesh2grid_graph()
        else:
            # 直接构建图数据
            # 图结构信息
            self.mesh2mesh_src_index = mesh2mesh_src_index
            self.mesh2mesh_dst_index = mesh2mesh_dst_index
            self.grid2mesh_src_index = grid2mesh_src_index
            self.grid2mesh_dst_index = grid2mesh_dst_index
            self.mesh2grid_src_index = mesh2grid_src_index
            self.mesh2grid_dst_index = mesh2grid_dst_index

            self.mesh_num_nodes = mesh_num_nodes
            self.grid_num_nodes = grid_num_nodes

            self.mesh_num_edges = mesh_num_edges
            self.grid2mesh_num_edges = grid2mesh_num_edges
            self.mesh2grid_num_edges = mesh2grid_num_edges

            # 图特征信息
            self.grid_node_feat = grid_node_feat
            self.mesh_node_feat = mesh_node_feat
            self.mesh_edge_feat = mesh_edge_feat
            self.grid2mesh_edge_feat = grid2mesh_edge_feat
            self.mesh2grid_edge_feat = mesh2grid_edge_feat

    def update(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError

    @property
    def finest_mesh(self):
        return self.meshes[-1]

    def init_mesh_properties(self):
        """Inits static properties that have to do with mesh nodes."""
        self.mesh_num_nodes = self.finest_mesh.vertices.shape[0]
        mesh_phi, mesh_theta = utils.cartesian_to_spherical(
            self.finest_mesh.vertices[:, 0],
            self.finest_mesh.vertices[:, 1],
            self.finest_mesh.vertices[:, 2],
        )
        (mesh_nodes_lat, mesh_nodes_lon) = utils.spherical_to_lat_lon(
            phi=mesh_phi,
            theta=mesh_theta,
        )
        # Convert to f32 to ensure the lat/lon features aren't in f64.
        self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
        self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """Inits static properties that have to do with grid nodes."""
        self._grid_lat = grid_lat.astype(np.float32)
        self._grid_lon = grid_lon.astype(np.float32)
        # Initialized the counters.
        self.grid_num_nodes = grid_lat.shape[0] * grid_lon.shape[0]

        # Initialize lat and lon for the grid.
        grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
        self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

    def _init_grid2mesh_graph(self):
        """Build Grid2Mesh graph."""

        # Create some edges according to distance between mesh and grid nodes.
        assert self._grid_lat is not None and self._grid_lon is not None
        (grid_indices, mesh_indices) = radius_query_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self.finest_mesh,
            radius=self.query_radius,
        )

        # Edges sending info from grid to mesh.
        senders = grid_indices
        receivers = mesh_indices

        # Precompute structural node and edge features according to config options.
        # Structural features are those that depend on the fixed values of the
        # latitude and longitudes of the nodes.
        (
            senders_node_features,
            _,
            edge_features,
        ) = utils.get_bipartite_graph_spatial_features(
            senders_node_lat=self._grid_nodes_lat,
            senders_node_lon=self._grid_nodes_lon,
            receivers_node_lat=self._mesh_nodes_lat,
            receivers_node_lon=self._mesh_nodes_lon,
            senders=senders,
            receivers=receivers,
            edge_normalization_factor=None,
            **self._spatial_features_kwargs,
        )

        self.grid_node_feat = np.expand_dims(senders_node_features, axis=1)

        self.grid2mesh_src_index = senders
        self.grid2mesh_dst_index = receivers
        self.grid2mesh_edge_feat = np.expand_dims(edge_features, axis=1)
        self.grid2mesh_num_edges = len(edge_features)

    def _init_mesh_graph(self):
        """Build Mesh graph."""
        merged_mesh = merge_meshes(self.meshes)
        # Work simply on the mesh edges.
        senders, receivers = faces_to_edges(merged_mesh.faces)
        # Precompute structural node and edge features according to config options.
        # Structural features are those that depend on the fixed values of the
        # latitude and longitudes of the nodes.
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        node_features, edge_features = utils.get_graph_spatial_features(
            node_lat=self._mesh_nodes_lat,
            node_lon=self._mesh_nodes_lon,
            senders=senders,
            receivers=receivers,
            **self._spatial_features_kwargs,
        )

        self.mesh_node_feat = np.expand_dims(node_features, axis=1)
        self.mesh2mesh_src_index = senders
        self.mesh2mesh_dst_index = receivers
        self.mesh_edge_feat = np.expand_dims(edge_features, axis=1)
        self.mesh_num_edges = len(edge_features)

    def _init_mesh2grid_graph(self):
        """Build Mesh2Grid graph."""

        # Create some edges according to how the grid nodes are contained by
        # mesh triangles.
        (grid_indices, mesh_indices) = in_mesh_triangle_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self.finest_mesh,
        )

        # Edges sending info from mesh to grid.
        senders = mesh_indices
        receivers = grid_indices

        # Precompute structural node and edge features according to config options.
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        (_, _, edge_features) = utils.get_bipartite_graph_spatial_features(
            senders_node_lat=self._mesh_nodes_lat,
            senders_node_lon=self._mesh_nodes_lon,
            receivers_node_lat=self._grid_nodes_lat,
            receivers_node_lon=self._grid_nodes_lon,
            senders=senders,
            receivers=receivers,
            edge_normalization_factor=self._mesh2grid_edge_normalization_factor,
            **self._spatial_features_kwargs,
        )

        self.mesh2grid_src_index = senders
        self.mesh2grid_dst_index = receivers
        self.mesh2grid_edge_feat = np.expand_dims(edge_features, axis=1)
        self.mesh2grid_num_edges = len(edge_features)

    @staticmethod
    def _get_max_edge_distance(mesh):
        senders, receivers = faces_to_edges(mesh.faces)
        edge_distances = np.linalg.norm(
            mesh.vertices[senders] - mesh.vertices[receivers], axis=-1
        )
        return edge_distances.max()

    def grid_node_outputs_to_prediction(
        self,
        grid_node_outputs: np.ndarray,
        targets_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """[num_grid_nodes, batch, num_outputs] -> xarray."""
        # numpy array with shape [lat_lon_node, batch, channels]
        # to xarray `DataArray` (batch, lat, lon, channels)
        assert self._grid_lat is not None and self._grid_lon is not None
        grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
        grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
            grid_shape + grid_node_outputs.shape[1:]
        )
        dims = ("lat", "lon", "batch", "channels")
        grid_xarray_lat_lon_leading = xarray.DataArray(
            data=grid_outputs_lat_lon_leading, dims=dims
        )
        grid_xarray = utils.restore_leading_axes(grid_xarray_lat_lon_leading)

        # xarray `DataArray` (batch, lat, lon, channels)
        # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
        return stacked_to_dataset(grid_xarray.variable, targets_template)


class TriangularMesh(typing.NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray


def merge_meshes(mesh_list: typing.Sequence[TriangularMesh]) -> TriangularMesh:
    for mesh_i, mesh_ip1 in itertools.pairwise(mesh_list):
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )


def get_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    vertices = []
    for c1, c2 in itertools.product([1.0, -1.0], [phi, -phi]):
        vertices.append((c1, c2, 0.0))
        vertices.append((0.0, c1, c2))
        vertices.append((c2, 0.0, c1))

    vertices = np.array(vertices, dtype=np.float32)
    vertices /= np.linalg.norm([1.0, phi])

    faces = [
        (0, 1, 2),
        (0, 6, 1),
        (8, 0, 2),
        (8, 4, 0),
        (3, 8, 2),
        (3, 2, 7),
        (7, 2, 1),
        (0, 4, 6),
        (4, 11, 6),
        (6, 11, 5),
        (1, 5, 7),
        (4, 10, 11),
        (4, 8, 10),
        (10, 8, 3),
        (10, 3, 9),
        (11, 10, 9),
        (11, 9, 5),
        (5, 9, 7),
        (9, 3, 7),
        (1, 6, 5),
    ]

    angle_between_faces = 2 * np.arcsin(phi / np.sqrt(3))
    rotation_angle = (np.pi - angle_between_faces) / 2
    rotation = scipy.spatial.transform.Rotation.from_euler(
        seq="y", angles=rotation_angle
    )
    rotation_matrix = rotation.as_matrix()
    vertices = np.dot(vertices, rotation_matrix)

    return TriangularMesh(
        vertices=vertices.astype(np.float32), faces=np.array(faces, dtype=np.int32)
    )


def get_hierarchy_of_triangular_meshes_for_sphere(
    splits: int,
) -> typing.List[TriangularMesh]:
    current_mesh = get_icosahedron()
    output_meshes = [current_mesh]
    for _ in range(splits):
        current_mesh = _two_split_unit_sphere_triangle_faces(current_mesh)
        output_meshes.append(current_mesh)
    return output_meshes


def _two_split_unit_sphere_triangle_faces(
    triangular_mesh: TriangularMesh,
) -> TriangularMesh:
    """Splits each triangular face into 4 triangles keeping the orientation."""
    new_vertices_builder = _ChildVerticesBuilder(triangular_mesh.vertices)

    new_faces = []
    for ind1, ind2, ind3 in triangular_mesh.faces:
        ind12 = new_vertices_builder.get_new_child_vertex_index((ind1, ind2))
        ind23 = new_vertices_builder.get_new_child_vertex_index((ind2, ind3))
        ind31 = new_vertices_builder.get_new_child_vertex_index((ind3, ind1))
        new_faces.extend(
            [
                [ind1, ind12, ind31],  # 1
                [ind12, ind2, ind23],  # 2
                [ind31, ind23, ind3],  # 3
                [ind12, ind23, ind31],  # 4
            ]
        )
    return TriangularMesh(
        vertices=new_vertices_builder.get_all_vertices(),
        faces=np.array(new_faces, dtype=np.int32),
    )


class _ChildVerticesBuilder(object):
    """Bookkeeping of new child vertices added to an existing set of vertices."""

    def __init__(self, parent_vertices):
        self._child_vertices_index_mapping = {}
        self._parent_vertices = parent_vertices
        # We start with all previous vertices.
        self._all_vertices_list = list(parent_vertices)

    def _get_child_vertex_key(self, parent_vertex_indices):
        return tuple(sorted(parent_vertex_indices))

    def _create_child_vertex(self, parent_vertex_indices):
        """Creates a new vertex."""
        # Position for new vertex is the middle point, between the parent points,
        # projected to unit sphere.
        child_vertex_position = self._parent_vertices[list(parent_vertex_indices)].mean(
            0
        )
        child_vertex_position /= np.linalg.norm(child_vertex_position)

        # Add the vertex to the output list. The index for this new vertex will
        # match the length of the list before adding it.
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        self._child_vertices_index_mapping[child_vertex_key] = len(
            self._all_vertices_list
        )
        self._all_vertices_list.append(child_vertex_position)

    def get_new_child_vertex_index(self, parent_vertex_indices):
        """Returns index for a child vertex, creating it if necessary."""
        # Get the key to see if we already have a new vertex in the middle.
        child_vertex_key = self._get_child_vertex_key(parent_vertex_indices)
        if child_vertex_key not in self._child_vertices_index_mapping:
            self._create_child_vertex(parent_vertex_indices)
        return self._child_vertices_index_mapping[child_vertex_key]

    def get_all_vertices(self):
        """Returns an array with old vertices."""
        return np.array(self._all_vertices_list)


def faces_to_edges(faces: np.ndarray):
    """Transforms polygonal faces to sender and receiver indices.

    It does so by transforming every face into N_i edges. Such if the triangular
    face has indices [0, 1, 2], three edges are added 0->1, 1->2, and 2->0.

    If all faces have consistent orientation, and the surface represented by the
    faces is closed, then every edge in a polygon with a certain orientation
    is also part of another polygon with the opposite orientation. In this
    situation, the edges returned by the method are always bidirectional.

    Args:
        faces: Integer array of shape [num_faces, 3]. Contains node indices
            adjacent to each face.
    Returns:
        Tuple with sender/receiver indices, each of shape [num_edges=num_faces*3].

    """
    assert faces.ndim == 2
    assert faces.shape[-1] == 3
    senders = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    receivers = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    return senders, receivers


def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray
) -> np.ndarray:
    """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
    # Convert to spherical coordinates phi and theta defined in the grid.
    # Each [num_latitude_points, num_longitude_points]
    phi_grid, theta_grid = np.meshgrid(
        np.deg2rad(grid_longitude), np.deg2rad(90 - grid_latitude)
    )

    # [num_latitude_points, num_longitude_points, 3]
    # Note this assumes unit radius, since for now we model the earth as a
    # sphere of unit radius, and keep any vertical dimension as a regular grid.
    return np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: TriangularMesh,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for radius query.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points]
      grid_longitude: Longitude values for the grid [num_lon_points]
      mesh: Mesh object.
      radius: Radius of connectivity in R3. for a sphere of unit radius.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges between the
      grid and the mesh such that the distances in a straight line (not geodesic)
      are smaller than or equal to `radius`.
      * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
      * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """

    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude
    ).reshape([-1, 3])

    # [num_mesh_points, 3]
    mesh_positions = mesh.vertices
    kd_tree = scipy.spatial.cKDTree(mesh_positions)

    # [num_grid_points, num_mesh_points_per_grid_point]
    # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
    # of arrays, rather than a 2d array.
    query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

    grid_edge_indices = []
    mesh_edge_indices = []
    for grid_index, mesh_neighbors in enumerate(query_indices):
        grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_indices.append(mesh_neighbors)

    # [num_edges]
    grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
    mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

    return grid_edge_indices, mesh_edge_indices


def in_mesh_triangle_indices(
    *, grid_latitude: np.ndarray, grid_longitude: np.ndarray, mesh: TriangularMesh
) -> tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for grid points contained in mesh triangles.

    Args:
      grid_latitude: Latitude values for the grid [num_lat_points]
      grid_longitude: Longitude values for the grid [num_lon_points]
      mesh: Mesh object.

    Returns:
      tuple with `grid_indices` and `mesh_indices` indicating edges between the
      grid and the mesh vertices of the triangle that contain each grid point.
      The number of edges is always num_lat_points * num_lon_points * 3
      * grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
      * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
    """

    # [num_grid_points=num_lat_points * num_lon_points, 3]
    grid_positions = _grid_lat_lon_to_coordinates(
        grid_latitude, grid_longitude
    ).reshape([-1, 3])

    mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # [num_grid_points] with mesh face indices for each grid point.
    _, _, query_face_indices = trimesh.proximity.closest_point(
        mesh_trimesh, grid_positions
    )

    # [num_grid_points, 3] with mesh node indices for each grid point.
    mesh_edge_indices = mesh.faces[query_face_indices]

    # [num_grid_points, 3] with grid node indices, where every row simply contains
    # the row (grid_point) index.
    grid_indices = np.arange(grid_positions.shape[0])
    grid_edge_indices = np.tile(grid_indices.reshape([-1, 1]), [1, 3])

    # Flatten to get a regular list.
    # [num_edges=num_grid_points*3]
    mesh_edge_indices = mesh_edge_indices.reshape([-1])
    grid_edge_indices = grid_edge_indices.reshape([-1])

    return grid_edge_indices, mesh_edge_indices


def convert_np_to_tensor(graph: GraphGridMesh):
    graph.mesh2mesh_src_index = paddle.to_tensor(
        graph.mesh2mesh_src_index, dtype=paddle.int64
    )
    graph.mesh2mesh_dst_index = paddle.to_tensor(
        graph.mesh2mesh_dst_index, dtype=paddle.int64
    )
    graph.grid2mesh_src_index = paddle.to_tensor(
        graph.grid2mesh_src_index, dtype=paddle.int64
    )
    graph.grid2mesh_dst_index = paddle.to_tensor(
        graph.grid2mesh_dst_index, dtype=paddle.int64
    )
    graph.mesh2grid_src_index = paddle.to_tensor(
        graph.mesh2grid_src_index, dtype=paddle.int64
    )
    graph.mesh2grid_dst_index = paddle.to_tensor(
        graph.mesh2grid_dst_index, dtype=paddle.int64
    )
    graph.grid_node_feat = paddle.to_tensor(
        graph.grid_node_feat, dtype=paddle.get_default_dtype()
    )
    graph.mesh_node_feat = paddle.to_tensor(
        graph.mesh_node_feat, dtype=paddle.get_default_dtype()
    )
    graph.mesh_edge_feat = paddle.to_tensor(
        graph.mesh_edge_feat, dtype=paddle.get_default_dtype()
    )
    graph.grid2mesh_edge_feat = paddle.to_tensor(
        graph.grid2mesh_edge_feat, dtype=paddle.get_default_dtype()
    )
    graph.mesh2grid_edge_feat = paddle.to_tensor(
        graph.mesh2grid_edge_feat, dtype=paddle.get_default_dtype()
    )
    return graph



_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Computes year progress for times in seconds.
    Args:
      seconds_since_epoch: Times in seconds since the "epoch" (the point at which
        UNIX time starts).
    Returns:
      Year progress normalized to be in the [0, 1) interval for each time point.
    """
    # Start with the pure integer division, and then float at the very end.
    # We will try to keep as much precision as possible.
    years_since_epoch = (
        seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
    )
    # Note depending on how these ops are down, we may end up with a "weak_type"
    # which can cause issues in subtle ways, and hard to track here.
    # In any case, casting to float32 should get rid of the weak type.
    # [0, 1.) Interval.
    return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Computes day progress for times in seconds at each longitude.
    Args:
      seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
        point at which UNIX time starts).
      longitude: 1D array of longitudes at which day progress is computed.
    Returns:
      2D array of day progress values normalized to be in the [0, 1) inverval
        for each time point at each longitude.
    """
    # [0.0, 1.0) Interval.
    day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY
    # Offset the day progress to the longitude of each point on Earth.
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(
        day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
    )
    return day_progress.astype(np.float32)


def datetime_features(seconds_since_epoch, longitude_offsets):
    year_progress = get_year_progress(seconds_since_epoch)
    day_progress = get_day_progress(seconds_since_epoch, longitude_offsets)
    year_progress_phase = year_progress * (2 * np.pi)
    day_progress_phase = day_progress * (2 * np.pi)
    returned_data = {
        "year_progress_sin": np.sin(year_progress_phase),
        "year_progress_cos": np.cos(year_progress_phase),
        "day_progress_sin": np.sin(day_progress_phase),
        "day_progress_cos": np.cos(day_progress_phase),
    }
    return returned_data


def add_var_into_nc_dataset(
    nc_dataset,
    var_name,
    var_value,
    var_dims=(
        "batch",
        "time",
    ),
):
    new_var = nc_dataset.createVariable(var_name, "f8", var_dims)
    new_var[:] = var_value
    return nc_dataset


def extract_input_target_times(
    dataset: xarray.Dataset,
    input_duration,
    target_lead_times,
):
    (target_lead_times, target_duration) = _process_target_lead_times_and_get_duration(
        target_lead_times
    )

    # Shift the coordinates for the time axis so that a timedelta of zero
    # corresponds to the forecast reference time. That is, the final timestep
    # that's available as input to the forecast, with all following timesteps
    # forming the target period which needs to be predicted.
    # This means the time coordinates are now forecast lead times.
    time = dataset.coords["time"]
    dataset = dataset.assign_coords(time=time + target_duration - time[-1])

    # Slice out targets:
    targets = dataset.sel({"time": target_lead_times})

    input_duration = pd.Timedelta(input_duration)
    # Both endpoints are inclusive with label-based slicing, so we offset by a
    # small epsilon to make one of the endpoints non-inclusive:
    zero = pd.Timedelta(0)
    epsilon = pd.Timedelta(1, "ns")
    inputs = dataset.sel({"time": slice(-input_duration + epsilon, zero)})
    return inputs, targets


def _process_target_lead_times_and_get_duration(target_lead_times):
    """Returns the minimum duration for the target lead times."""
    if isinstance(target_lead_times, slice):
        # A slice of lead times. xarray already accepts timedelta-like values for
        # the begin/end/step of the slice.
        if target_lead_times.start is None:
            # If the start isn't specified, we assume it starts at the next timestep
            # after lead time 0 (lead time 0 is the final input timestep):
            target_lead_times = slice(
                pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step
            )
        target_duration = pd.Timedelta(target_lead_times.stop)
    else:
        if not isinstance(target_lead_times, (list, tuple, set)):
            # A single lead time, which we wrap as a length-1 array to ensure there
            # still remains a time dimension (here of length 1) for consistency.
            target_lead_times = [target_lead_times]

        # A list of multiple (not necessarily contiguous) lead times:
        target_lead_times = [pd.Timedelta(x) for x in target_lead_times]
        target_lead_times.sort()
        target_duration = target_lead_times[-1]
    return target_lead_times, target_duration


def variable_to_stacked(
    variable: xarray.Variable,
    sizes,
    preserved_dims=("batch", "lat", "lon"),
) -> xarray.Variable:
    """Converts an xarray.Variable to preserved_dims + ("channels",).

    Any dimensions other than those included in preserved_dims get stacked into a
    final "channels" dimension. If any of the preserved_dims are missing then they
    are added, with the data broadcast/tiled to match the sizes specified in
    `sizes`.

    Args:
      variable: An xarray.Variable.
      sizes: Mapping including sizes for any dimensions which are not present in
        `variable` but are needed for the output. This may be needed for example
        for a static variable with only ("lat", "lon") dims, or if you want to
        encode just the latitude coordinates (a variable with dims ("lat",)).
      preserved_dims: dimensions of variable to not be folded in channels.

    Returns:
      An xarray.Variable with dimensions preserved_dims + ("channels",).
    """
    stack_to_channels_dims = [d for d in variable.dims if d not in preserved_dims]
    if stack_to_channels_dims:
        variable = variable.stack(channels=stack_to_channels_dims)
    dims = {dim: variable.sizes.get(dim) or sizes[dim] for dim in preserved_dims}
    dims["channels"] = variable.sizes.get("channels", 1)
    return variable.set_dims(dims)


def dataset_to_stacked(
    dataset: xarray.Dataset,
    sizes=None,
    preserved_dims=("batch", "lat", "lon"),
) -> xarray.DataArray:
    """Converts an xarray.Dataset to a single stacked array.

    This takes each consistuent data_var, converts it into BHWC layout
    using `variable_to_stacked`, then concats them all along the channels axis.

    Args:
      dataset: An xarray.Dataset.
      sizes: Mapping including sizes for any dimensions which are not present in
        the `dataset` but are needed for the output. See variable_to_stacked.
      preserved_dims: dimensions from the dataset that should not be folded in
        the predictions channels.

    Returns:
      An xarray.DataArray with dimensions preserved_dims + ("channels",).
      Existing coordinates for preserved_dims axes will be preserved, however
      there will be no coordinates for "channels".
    """
    data_vars = [
        variable_to_stacked(
            dataset.variables[name], sizes or dataset.sizes, preserved_dims
        )
        for name in sorted(dataset.data_vars.keys())
    ]
    coords = {
        dim: coord for dim, coord in dataset.coords.items() if dim in preserved_dims
    }
    return xarray.DataArray(
        data=xarray.Variable.concat(data_vars, dim="channels"), coords=coords
    )


def stacked_to_dataset(
    stacked_array: xarray.Variable,
    template_dataset: xarray.Dataset,
    preserved_dims: typing.Tuple[str, ...] = ("batch", "lat", "lon"),
) -> xarray.Dataset:
    """The inverse of dataset_to_stacked.

    Requires a template dataset to demonstrate the variables/shapes/coordinates
    required.
    All variables must have preserved_dims dimensions.

    Args:
      stacked_array: Data in BHWC layout, encoded the same as dataset_to_stacked
        would if it was asked to encode `template_dataset`.
      template_dataset: A template Dataset (or other mapping of DataArrays)
        demonstrating the shape of output required (variables, shapes,
        coordinates etc).
      preserved_dims: dimensions from the target_template that were not folded in
        the predictions channels. The preserved_dims need to be a subset of the
        dims of all the variables of template_dataset.

    Returns:
      An xarray.Dataset (or other mapping of DataArrays) with the same shape and
      type as template_dataset.
    """
    unstack_from_channels_sizes = {}
    var_names = sorted(template_dataset.keys())
    for name in var_names:
        template_var = template_dataset[name]
        if not all(dim in template_var.dims for dim in preserved_dims):
            raise ValueError(
                f"stacked_to_dataset requires all Variables to have {preserved_dims} "
                f"dimensions, but found only {template_var.dims}."
            )
        unstack_from_channels_sizes[name] = {
            dim: size
            for dim, size in template_var.sizes.items()
            if dim not in preserved_dims
        }

    channels = {
        name: np.prod(list(unstack_sizes.values()), dtype=np.int64)
        for name, unstack_sizes in unstack_from_channels_sizes.items()
    }
    total_expected_channels = sum(channels.values())
    found_channels = stacked_array.sizes["channels"]
    if total_expected_channels != found_channels:
        raise ValueError(
            f"Expected {total_expected_channels} channels but found "
            f"{found_channels}, when trying to convert a stacked array of shape "
            f"{stacked_array.sizes} to a dataset of shape {template_dataset}."
        )

    data_vars = {}
    index = 0
    for name in var_names:
        template_var = template_dataset[name]
        var = stacked_array.isel({"channels": slice(index, index + channels[name])})
        index += channels[name]
        var = var.unstack({"channels": unstack_from_channels_sizes[name]})
        var = var.transpose(*template_var.dims)
        data_vars[name] = xarray.DataArray(
            data=var,
            coords=template_var.coords,
            # This might not always be the same as the name it's keyed under; it
            # will refer to the original variable name, whereas the key might be
            # some alias e.g. temperature_850 under which it should be logged:
            name=template_var.name,
        )
    return type(template_dataset)(
        data_vars
    )  # pytype:disable=not-callable,wrong-arg-count


class ERA5Data(io.Dataset):
    """
    This class is used to process ERA5 re-analyze data,
    and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training,
        evaluation or testing. Supports [“train”,“test”, “valid”].
        Default: 'train'.

    Examples:
        >>> from mindearth.data import Era5Data
        >>> data_params = {
        ...     'name': 'era5',
        ...     'root_dir': './dataset',
        ...     'w_size': 256
        ... }
        >>> dataset_generator = Era5Data(data_params)
    """

    # TODO: example should include all possible infos:
    #  data_frequency, patch/patch_size
    def __init__(self, config, data_type="train"):
        super().__init__()
        if config.type == "graphcast":
            self.input_variables = args.TASK_input_variables
            self.forcing_variables = args.TASK_forcing_variables
            self.target_variables = args.TASK_target_variables
            self.level_variables = args.PRESSURE_LEVELS[37]
        elif config.type == "graphcast_small":
            self.input_variables = args.TASK_13_input_variables
            self.forcing_variables = args.TASK_13_forcing_variables
            self.target_variables = args.TASK_13_target_variables
            self.level_variables = args.PRESSURE_LEVELS[13]
        elif config.type == "graphcast_operational":
            self.input_variables = args.TASK_13_PRECIP_OUT_input_variables
            self.forcing_variables = args.TASK_13_PRECIP_OUT_forcing_variables
            self.target_variables = args.TASK_13_PRECIP_OUT_target_variables
            self.level_variables = args.PRESSURE_LEVELS[13]

        # 数据
        nc_dataset = xarray.open_dataset(config.data_path)

        longitude_offsets = nc_dataset.coords["lon"].data
        second_since_epoch = (
            nc_dataset.coords["datetime"].data.astype("datetime64[s]").astype(np.int64)
        )
        datetime_feats = datetime_features(second_since_epoch, longitude_offsets)
        nc_dataset.update(
            {
                "year_progress_sin": xarray.Variable(
                    ("batch", "time"), datetime_feats["year_progress_sin"]
                ),
                "year_progress_cos": xarray.Variable(
                    ("batch", "time"), datetime_feats["year_progress_cos"]
                ),
                "day_progress_sin": xarray.Variable(
                    ("batch", "time", "lon"), datetime_feats["day_progress_sin"]
                ),
                "day_progress_cos": xarray.Variable(
                    ("batch", "time", "lon"), datetime_feats["day_progress_cos"]
                ),
            }
        )

        inputs, targets = extract_input_target_times(
            nc_dataset, input_duration="12h", target_lead_times="6h"
        )

        # 统计数据
        stddev_data = xarray.open_dataset(config.stddev_path).sel(
            level=list(self.level_variables)
        )
        stddev_diffs_data = xarray.open_dataset(config.stddev_diffs_path).sel(
            level=list(self.level_variables)
        )
        mean_data = xarray.open_dataset(config.mean_path).sel(
            level=list(self.level_variables)
        )

        missing_variables = set(self.target_variables) - set(self.input_variables)
        exist_variables = set(self.target_variables) - missing_variables
        targets_stddev = stddev_diffs_data[list(exist_variables)]
        target_mean = inputs[list(exist_variables)].isel(time=-1)
        if missing_variables:
            targets_stddev.update({var: stddev_data[var] for var in missing_variables})
            target_mean.update(
                {var: mean_data.variables[var] for var in missing_variables}
            )

        stacked_targets_stddev = dataset_to_stacked(targets_stddev, preserved_dims=())
        stacked_targets_mean = dataset_to_stacked(target_mean)
        stacked_targets_mean = stacked_targets_mean.transpose("lat", "lon", ...)

        # The forcing uses the same time coordinates as the target.
        inputs = inputs[list(self.input_variables)]
        forcings = targets[list(self.forcing_variables)]
        targets = targets[list(self.target_variables)]
        inputs = self.normalize(inputs, stddev_data, mean_data)
        forcings = self.normalize(forcings, stddev_data, mean_data)

        self.targets_template = targets

        stacked_inputs = dataset_to_stacked(inputs)
        stacked_forcings = dataset_to_stacked(forcings)
        stacked_targets = dataset_to_stacked(targets)
        stacked_inputs = xarray.concat(
            [stacked_inputs, stacked_forcings], dim="channels"
        )

        stacked_inputs = stacked_inputs.transpose("lat", "lon", ...)
        stacked_targets = stacked_targets.transpose("lat", "lon", ...)

        # 此处指定input数据为12h数据，target数据为6h数据
        lat_dim, lon_dim, batch_dim, feat_dim = stacked_inputs.shape
        stacked_inputs = stacked_inputs.data.reshape(lat_dim * lon_dim, batch_dim, -1)
        stacked_targets = stacked_targets.data.reshape(lat_dim * lon_dim, batch_dim, -1)
        self.stacked_targets_stddev = stacked_targets_stddev.data
        self.stacked_targets_mean = stacked_targets_mean.data.reshape(
            lat_dim * lon_dim, batch_dim, -1
        )

        self.input_data = []
        self.target_data = []

        graph_template_path = os.path.join(
            "data", "template_graph", f"{config.type}.pkl"
        )
        if os.path.exists(graph_template_path):
            graph_template = pickle.load(open(graph_template_path, "rb"))
        else:
            graph_template = GraphGridMesh(config)

        graph = copy.deepcopy(graph_template)
        graph.grid_node_feat = np.concatenate(
            [stacked_inputs, graph.grid_node_feat], axis=-1
        )
        mesh_node_feat = np.zeros([graph.mesh_num_nodes, batch_dim, feat_dim])
        graph.mesh_node_feat = np.concatenate(
            [mesh_node_feat, graph.mesh_node_feat], axis=-1
        )

        self.input_data.append(graph)
        self.target_data.append(stacked_targets)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

    def normalize(self, inputs_data, stddev_data, mean_data):
        for name in list(inputs_data.keys()):
            inputs_data[name] = (inputs_data[name] - mean_data[name]) / stddev_data[
                name
            ]
        return inputs_data

    def denormalize(self, inputs_data):
        return inputs_data * self.stacked_targets_stddev + self.stacked_targets_mean

# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for building models."""

from typing import Optional
from typing import Tuple

import numpy as np
import scipy
import xarray


def get_graph_spatial_features(
    *,
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_relative_positions: bool,
    relative_longitude_local_coordinates: bool,
    relative_latitude_local_coordinates: bool,
    sine_cosine_encoding: bool = False,
    encoding_num_freqs: int = 10,
    encoding_multiplicative_factor: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes spatial features for the nodes.

    Args:
      node_lat: Latitudes in the [-90, 90] interval of shape [num_nodes]
      node_lon: Longitudes in the [0, 360] interval of shape [num_nodes]
      senders: Sender indices of shape [num_edges]
      receivers: Receiver indices of shape [num_edges]
      add_node_positions: Add unit norm absolute positions.
      add_node_latitude: Add a feature for latitude (cos(90 - lat))
          Note even if this is set to False, the model may be able to infer the
          longitude from relative features, unless
          `relative_latitude_local_coordinates` is also True, or if there is any
          bias on the relative edge sizes for different longitudes.
      add_node_longitude: Add features for longitude (cos(lon), sin(lon)).
          Note even if this is set to False, the model may be able to infer the
          longitude from relative features, unless
          `relative_longitude_local_coordinates` is also True, or if there is any
          bias on the relative edge sizes for different longitudes.
      add_relative_positions: Whether to relative positions in R3 to the edges.
      relative_longitude_local_coordinates: If True, relative positions are
          computed in a local space where the receiver is at 0 longitude.
      relative_latitude_local_coordinates: If True, relative positions are
          computed in a local space where the receiver is at 0 latitude.
      sine_cosine_encoding: If True, we will transform the node/edge features
          with sine and cosine functions, similar to NERF.
      encoding_num_freqs: frequency parameter
      encoding_multiplicative_factor: used for calculating the frequency.

    Returns:
      Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
      with node and edge features.

    """

    num_nodes = node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = node_lat.dtype
    node_phi, node_theta = lat_lon_deg_to_spherical(node_lat, node_lon)

    # Computing some node features.
    node_features = []
    if add_node_positions:
        # Already in [-1, 1.] range.
        node_features.extend(spherical_to_cartesian(node_phi, node_theta))

    if add_node_latitude:
        # Using the cos of theta.
        # From 1. (north pole) to -1 (south pole).
        node_features.append(np.cos(node_theta))

    if add_node_longitude:
        # Using the cos and sin, which is already normalized.
        node_features.append(np.cos(node_phi))
        node_features.append(np.sin(node_phi))

    if not node_features:
        node_features = np.zeros([num_nodes, 0], dtype=dtype)
    else:
        node_features = np.stack(node_features, axis=-1)

    # Computing some edge features.
    edge_features = []

    if add_relative_positions:

        relative_position = get_relative_position_in_receiver_local_coordinates(
            node_phi=node_phi,
            node_theta=node_theta,
            senders=senders,
            receivers=receivers,
            latitude_local_coordinates=relative_latitude_local_coordinates,
            longitude_local_coordinates=relative_longitude_local_coordinates,
        )

        # Note this is L2 distance in 3d space, rather than geodesic distance.
        relative_edge_distances = np.linalg.norm(
            relative_position, axis=-1, keepdims=True
        )

        # Normalize to the maximum edge distance. Note that we expect to always
        # have an edge that goes in the opposite direction of any given edge
        # so the distribution of relative positions should be symmetric around
        # zero. So by scaling by the maximum length, we expect all relative
        # positions to fall in the [-1., 1.] interval, and all relative distances
        # to fall in the [0., 1.] interval.
        max_edge_distance = relative_edge_distances.max()
        edge_features.append(relative_edge_distances / max_edge_distance)
        edge_features.append(relative_position / max_edge_distance)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    if sine_cosine_encoding:

        def sine_cosine_transform(x: np.ndarray) -> np.ndarray:
            freqs = encoding_multiplicative_factor ** np.arange(encoding_num_freqs)
            phases = freqs * x[..., None]
            x_sin = np.sin(phases)
            x_cos = np.cos(phases)
            x_cat = np.concatenate([x_sin, x_cos], axis=-1)
            return x_cat.reshape([x.shape[0], -1])

        node_features = sine_cosine_transform(node_features)
        edge_features = sine_cosine_transform(edge_features)

    return node_features, edge_features


def lat_lon_to_leading_axes(grid_xarray: xarray.DataArray) -> xarray.DataArray:
    """Reorders xarray so lat/lon axes come first."""
    # leading + ["lat", "lon"] + trailing
    # to
    # ["lat", "lon"] + leading + trailing
    return grid_xarray.transpose("lat", "lon", ...)


def restore_leading_axes(grid_xarray: xarray.DataArray) -> xarray.DataArray:
    """Reorders xarray so batch/time/level axes come first (if present)."""

    # ["lat", "lon"] + [(batch,) (time,) (level,)] + trailing
    # to
    # [(batch,) (time,) (level,)] + ["lat", "lon"] + trailing

    input_dims = list(grid_xarray.dims)
    output_dims = list(input_dims)
    for leading_key in ["level", "time", "batch"]:  # reverse order for insert
        if leading_key in input_dims:
            output_dims.remove(leading_key)
            output_dims.insert(0, leading_key)
    return grid_xarray.transpose(*output_dims)


def lat_lon_deg_to_spherical(
    node_lat: np.ndarray,
    node_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(node_lon)
    theta = np.deg2rad(90 - node_lat)
    return phi, theta


def spherical_to_lat_lon(
    phi: np.ndarray,
    theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.mod(np.rad2deg(phi), 360)
    lat = 90 - np.rad2deg(theta)
    return lat, lon


def cartesian_to_spherical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(y, x)
    with np.errstate(invalid="ignore"):  # circumventing b/253179568
        theta = np.arccos(z)  # Assuming unit radius.
    return phi, theta


def spherical_to_cartesian(
    phi: np.ndarray, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Assuming unit radius.
    return (np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta))


def get_relative_position_in_receiver_local_coordinates(
    node_phi: np.ndarray,
    node_theta: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    latitude_local_coordinates: bool,
    longitude_local_coordinates: bool,
) -> np.ndarray:
    """Returns relative position features for the edges.

    The relative positions will be computed in a rotated space for a local
    coordinate system as defined by the receiver. The relative positions are
    simply obtained by subtracting sender position minues receiver position in
    that local coordinate system after the rotation in R^3.

    Args:
      node_phi: [num_nodes] with polar angles.
      node_theta: [num_nodes] with azimuthal angles.
      senders: [num_edges] with indices.
      receivers: [num_edges] with indices.
      latitude_local_coordinates: Whether to rotate edges such that in the
          positions are computed such that the receiver is always at latitude 0.
      longitude_local_coordinates: Whether to rotate edges such that in the
          positions are computed such that the receiver is always at longitude 0.

    Returns:
      Array of relative positions in R3 [num_edges, 3]
    """

    node_pos = np.stack(spherical_to_cartesian(node_phi, node_theta), axis=-1)

    # No rotation in this case.
    if not (latitude_local_coordinates or longitude_local_coordinates):
        return node_pos[senders] - node_pos[receivers]

    # Get rotation matrices for the local space space for every node.
    rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=node_phi,
        reference_theta=node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
    )

    # Each edge will be rotated according to the rotation matrix of its receiver
    # node.
    edge_rotation_matrices = rotation_matrices[receivers]

    # Rotate all nodes to the rotated space of the corresponding edge.
    # Note for receivers we can also do the matmul first and the gather second:
    # ```
    # receiver_pos_in_rotated_space = rotate_with_matrices(
    #    rotation_matrices, node_pos)[receivers]
    # ```
    # which is more efficient, however, we do gather first to keep it more
    # symmetric with the sender computation.
    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[receivers]
    )
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos[senders]
    )
    # Note, here, that because the rotated space is chosen according to the
    # receiver, if:
    # * latitude_local_coordinates = True: latitude for the receivers will be
    #   0, that is the z coordinate will always be 0.
    # * longitude_local_coordinates = True: longitude for the receivers will be
    #   0, that is the y coordinate will be 0.

    # Now we can just subtract.
    # Note we are rotating to a local coordinate system, where the y-z axes are
    # parallel to a tangent plane to the sphere, but still remain in a 3d space.
    # Note that if both `latitude_local_coordinates` and
    # `longitude_local_coordinates` are True, and edges are short,
    # then the difference in x coordinate between sender and receiver
    # should be small, so we could consider dropping the new x coordinate if
    # we wanted to the tangent plane, however in doing so
    # we would lose information about the curvature of the mesh, which may be
    # important for very coarse meshes.
    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space


def get_rotation_matrices_to_local_coordinates(
    reference_phi: np.ndarray,
    reference_theta: np.ndarray,
    rotate_latitude: bool,
    rotate_longitude: bool,
) -> np.ndarray:
    """Returns a rotation matrix to rotate to a point based on a reference vector.

    The rotation matrix is build such that, a vector in the
    same coordinate system at the reference point that points towards the pole
    before the rotation, continues to point towards the pole after the rotation.

    Args:
      reference_phi: [leading_axis] Polar angles of the reference.
      reference_theta: [leading_axis] Azimuthal angles of the reference.
      rotate_latitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero latitude.
      rotate_longitude: Whether to produce a rotation matrix that would rotate
          R^3 vectors to zero longitude.

    Returns:
      Matrices of shape [leading_axis] such that when applied to the reference
          position with `rotate_with_matrices(rotation_matrices, reference_pos)`

          * phi goes to 0. if "rotate_longitude" is True.

          * theta goes to np.pi / 2 if "rotate_latitude" is True.

          The rotation consists of:
          * rotate_latitude = False, rotate_longitude = True:
              Latitude preserving rotation.
          * rotate_latitude = True, rotate_longitude = True:
              Latitude preserving rotation, followed by longitude preserving
              rotation.
          * rotate_latitude = True, rotate_longitude = False:
              Latitude preserving rotation, followed by longitude preserving
              rotation, and the inverse of the latitude preserving rotation. Note
              this is computationally different from rotating the longitude only
              and is. We do it like this, so the polar geodesic curve, continues
              to be aligned with one of the axis after the rotation.

    """

    if rotate_longitude and rotate_latitude:

        # We first rotate around the z axis "minus the azimuthal angle", to get the
        # point with zero longitude
        azimuthal_rotation = -reference_phi

        # One then we will do a polar rotation (which can be done along the y
        # axis now that we are at longitude 0.), "minus the polar angle plus 2pi"
        # to get the point with zero latitude.
        polar_rotation = -reference_theta + np.pi / 2

        return scipy.spatial.transform.Rotation.from_euler(
            "zy", np.stack([azimuthal_rotation, polar_rotation], axis=1)
        ).as_matrix()
    elif rotate_longitude:
        # Just like the previous case, but applying only the azimuthal rotation.
        azimuthal_rotation = -reference_phi
        return scipy.spatial.transform.Rotation.from_euler(
            "z", -reference_phi
        ).as_matrix()
    elif rotate_latitude:
        # Just like the first case, but after doing the polar rotation, undoing
        # the azimuthal rotation.
        azimuthal_rotation = -reference_phi
        polar_rotation = -reference_theta + np.pi / 2

        return scipy.spatial.transform.Rotation.from_euler(
            "zyz",
            np.stack([azimuthal_rotation, polar_rotation, -azimuthal_rotation], axis=1),
        ).as_matrix()
    else:
        raise ValueError("At least one of longitude and latitude should be rotated.")


def rotate_with_matrices(
    rotation_matrices: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    return np.einsum("bji,bi->bj", rotation_matrices, positions)


def get_bipartite_graph_spatial_features(
    *,
    senders_node_lat: np.ndarray,
    senders_node_lon: np.ndarray,
    senders: np.ndarray,
    receivers_node_lat: np.ndarray,
    receivers_node_lon: np.ndarray,
    receivers: np.ndarray,
    add_node_positions: bool,
    add_node_latitude: bool,
    add_node_longitude: bool,
    add_relative_positions: bool,
    edge_normalization_factor: Optional[float] = None,
    relative_longitude_local_coordinates: bool,
    relative_latitude_local_coordinates: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes spatial features for the nodes.

    This function is almost identical to `get_graph_spatial_features`. The only
    difference is that sender nodes and receiver nodes can be in different arrays.
    This is necessary to enable combination with typed Graph.

    Args:
      senders_node_lat: Latitudes in the [-90, 90] interval of shape
        [num_sender_nodes]
      senders_node_lon: Longitudes in the [0, 360] interval of shape
        [num_sender_nodes]
      senders: Sender indices of shape [num_edges], indices in [0,
        num_sender_nodes)
      receivers_node_lat: Latitudes in the [-90, 90] interval of shape
        [num_receiver_nodes]
      receivers_node_lon: Longitudes in the [0, 360] interval of shape
        [num_receiver_nodes]
      receivers: Receiver indices of shape [num_edges], indices in [0,
        num_receiver_nodes)
      add_node_positions: Add unit norm absolute positions.
      add_node_latitude: Add a feature for latitude (cos(90 - lat)) Note even if
        this is set to False, the model may be able to infer the longitude from
        relative features, unless `relative_latitude_local_coordinates` is also
        True, or if there is any bias on the relative edge sizes for different
        longitudes.
      add_node_longitude: Add features for longitude (cos(lon), sin(lon)). Note
        even if this is set to False, the model may be able to infer the longitude
        from relative features, unless `relative_longitude_local_coordinates` is
        also True, or if there is any bias on the relative edge sizes for
        different longitudes.
      add_relative_positions: Whether to relative positions in R3 to the edges.
      edge_normalization_factor: Allows explicitly controlling edge normalization.
        If None, defaults to max edge length. This supports using pre-trained
        model weights with a different graph structure to what it was trained on.
      relative_longitude_local_coordinates: If True, relative positions are
        computed in a local space where the receiver is at 0 longitude.
      relative_latitude_local_coordinates: If True, relative positions are
        computed in a local space where the receiver is at 0 latitude.

    Returns:
      Arrays of shape: [num_nodes, num_features] and [num_edges, num_features].
      with node and edge features.

    """

    num_senders = senders_node_lat.shape[0]
    num_receivers = receivers_node_lat.shape[0]
    num_edges = senders.shape[0]
    dtype = senders_node_lat.dtype
    assert receivers_node_lat.dtype == dtype
    senders_node_phi, senders_node_theta = lat_lon_deg_to_spherical(
        senders_node_lat, senders_node_lon
    )
    receivers_node_phi, receivers_node_theta = lat_lon_deg_to_spherical(
        receivers_node_lat, receivers_node_lon
    )

    # Computing some node features.
    senders_node_features = []
    receivers_node_features = []
    if add_node_positions:
        # Already in [-1, 1.] range.
        senders_node_features.extend(
            spherical_to_cartesian(senders_node_phi, senders_node_theta)
        )
        receivers_node_features.extend(
            spherical_to_cartesian(receivers_node_phi, receivers_node_theta)
        )

    if add_node_latitude:
        # Using the cos of theta.
        # From 1. (north pole) to -1 (south pole).
        senders_node_features.append(np.cos(senders_node_theta))
        receivers_node_features.append(np.cos(receivers_node_theta))

    if add_node_longitude:
        # Using the cos and sin, which is already normalized.
        senders_node_features.append(np.cos(senders_node_phi))
        senders_node_features.append(np.sin(senders_node_phi))

        receivers_node_features.append(np.cos(receivers_node_phi))
        receivers_node_features.append(np.sin(receivers_node_phi))

    if not senders_node_features:
        senders_node_features = np.zeros([num_senders, 0], dtype=dtype)
        receivers_node_features = np.zeros([num_receivers, 0], dtype=dtype)
    else:
        senders_node_features = np.stack(senders_node_features, axis=-1)
        receivers_node_features = np.stack(receivers_node_features, axis=-1)

    # Computing some edge features.
    edge_features = []

    if add_relative_positions:

        relative_position = (
            get_bipartite_relative_position_in_receiver_local_coordinates(
                senders_node_phi=senders_node_phi,
                senders_node_theta=senders_node_theta,
                receivers_node_phi=receivers_node_phi,
                receivers_node_theta=receivers_node_theta,
                senders=senders,
                receivers=receivers,
                latitude_local_coordinates=relative_latitude_local_coordinates,
                longitude_local_coordinates=relative_longitude_local_coordinates,
            )
        )

        # Note this is L2 distance in 3d space, rather than geodesic distance.
        relative_edge_distances = np.linalg.norm(
            relative_position, axis=-1, keepdims=True
        )

        if edge_normalization_factor is None:
            # Normalize to the maximum edge distance. Note that we expect to always
            # have an edge that goes in the opposite direction of any given edge
            # so the distribution of relative positions should be symmetric around
            # zero. So by scaling by the maximum length, we expect all relative
            # positions to fall in the [-1., 1.] interval, and all relative distances
            # to fall in the [0., 1.] interval.
            edge_normalization_factor = relative_edge_distances.max()

        edge_features.append(relative_edge_distances / edge_normalization_factor)
        edge_features.append(relative_position / edge_normalization_factor)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    return senders_node_features, receivers_node_features, edge_features


def get_bipartite_relative_position_in_receiver_local_coordinates(
    senders_node_phi: np.ndarray,
    senders_node_theta: np.ndarray,
    senders: np.ndarray,
    receivers_node_phi: np.ndarray,
    receivers_node_theta: np.ndarray,
    receivers: np.ndarray,
    latitude_local_coordinates: bool,
    longitude_local_coordinates: bool,
) -> np.ndarray:
    """Returns relative position features for the edges.

    This function is equivalent to
    `get_relative_position_in_receiver_local_coordinates`, but adapted to work
    with bipartite typed graphs.

    The relative positions will be computed in a rotated space for a local
    coordinate system as defined by the receiver. The relative positions are
    simply obtained by subtracting sender position minues receiver position in
    that local coordinate system after the rotation in R^3.

    Args:
      senders_node_phi: [num_sender_nodes] with polar angles.
      senders_node_theta: [num_sender_nodes] with azimuthal angles.
      senders: [num_edges] with indices into sender nodes.
      receivers_node_phi: [num_sender_nodes] with polar angles.
      receivers_node_theta: [num_sender_nodes] with azimuthal angles.
      receivers: [num_edges] with indices into receiver nodes.
      latitude_local_coordinates: Whether to rotate edges such that in the
        positions are computed such that the receiver is always at latitude 0.
      longitude_local_coordinates: Whether to rotate edges such that in the
        positions are computed such that the receiver is always at longitude 0.

    Returns:
      Array of relative positions in R3 [num_edges, 3]
    """

    senders_node_pos = np.stack(
        spherical_to_cartesian(senders_node_phi, senders_node_theta), axis=-1
    )

    receivers_node_pos = np.stack(
        spherical_to_cartesian(receivers_node_phi, receivers_node_theta), axis=-1
    )

    # No rotation in this case.
    if not (latitude_local_coordinates or longitude_local_coordinates):
        return senders_node_pos[senders] - receivers_node_pos[receivers]

    # Get rotation matrices for the local space space for every receiver node.
    receiver_rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=receivers_node_phi,
        reference_theta=receivers_node_theta,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates,
    )

    # Each edge will be rotated according to the rotation matrix of its receiver
    # node.
    edge_rotation_matrices = receiver_rotation_matrices[receivers]

    # Rotate all nodes to the rotated space of the corresponding edge.
    # Note for receivers we can also do the matmul first and the gather second:
    # ```
    # receiver_pos_in_rotated_space = rotate_with_matrices(
    #    rotation_matrices, node_pos)[receivers]
    # ```
    # which is more efficient, however, we do gather first to keep it more
    # symmetric with the sender computation.
    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, receivers_node_pos[receivers]
    )
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, senders_node_pos[senders]
    )
    # Note, here, that because the rotated space is chosen according to the
    # receiver, if:
    # * latitude_local_coordinates = True: latitude for the receivers will be
    #   0, that is the z coordinate will always be 0.
    # * longitude_local_coordinates = True: longitude for the receivers will be
    #   0, that is the y coordinate will be 0.

    # Now we can just subtract.
    # Note we are rotating to a local coordinate system, where the y-z axes are
    # parallel to a tangent plane to the sphere, but still remain in a 3d space.
    # Note that if both `latitude_local_coordinates` and
    # `longitude_local_coordinates` are True, and edges are short,
    # then the difference in x coordinate between sender and receiver
    # should be small, so we could consider dropping the new x coordinate if
    # we wanted to the tangent plane, however in doing so
    # we would lose information about the curvature of the mesh, which may be
    # important for very coarse meshes.
    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space

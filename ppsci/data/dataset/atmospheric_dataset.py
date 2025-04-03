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

from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import paddle
import pandas as pd
import scipy
from paddle import io

try:
    import trimesh
    import xarray
except ModuleNotFoundError:
    pass

# https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
PRESSURE_LEVELS_ERA5_37 = (
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
)

# https://www.ecmwf.int/en/forecasts/datasets/set-i
PRESSURE_LEVELS_HRES_25 = (
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
    850,
    900,
    925,
    950,
    1000,
)

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203
PRESSURE_LEVELS_WEATHERBENCH_13 = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

PRESSURE_LEVELS = {
    13: PRESSURE_LEVELS_WEATHERBENCH_13,
    25: PRESSURE_LEVELS_HRES_25,
    37: PRESSURE_LEVELS_ERA5_37,
}


TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)
EXTERNAL_FORCING_VARS = ("toa_incident_solar_radiation",)
GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)
FORCING_VARS = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS
STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)

TASK_input_variables = (
    TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS + STATIC_VARS
)
TASK_target_variables = TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS
TASK_forcing_variables = FORCING_VARS
TASK_pressure_levels = PRESSURE_LEVELS_ERA5_37
TASK_input_duration = ("12h",)

TASK_13_input_variables = (
    TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS + STATIC_VARS
)
TASK_13_target_variables = TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS
TASK_13_forcing_variables = FORCING_VARS
TASK_13_pressure_levels = PRESSURE_LEVELS_WEATHERBENCH_13
TASK_13_input_duration = ("12h",)


TASK_13_PRECIP_OUT_input_variables = (
    TARGET_SURFACE_NO_PRECIP_VARS + TARGET_ATMOSPHERIC_VARS + FORCING_VARS + STATIC_VARS
)
TASK_13_PRECIP_OUT_target_variables = TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS
TASK_13_PRECIP_OUT_forcing_variables = FORCING_VARS
TASK_13_PRECIP_OUT_pressure_levels = PRESSURE_LEVELS_WEATHERBENCH_13
TASK_13_PRECIP_OUT_input_duration = ("12h",)

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"


def stacked_to_dataset(
    stacked_array: "xarray.Variable",
    template_dataset: "xarray.Dataset",
    preserved_dims: Tuple[str, ...] = ("batch", "lat", "lon"),
) -> "xarray.Dataset":
    """The inverse of dataset_to_stacked.

    Requires a template dataset to demonstrate the variables/shapes/coordinates
    required.
    All variables must have preserved_dims dimensions.

    Args:
        stacked_array: Data in BHWC layout, encoded the same as dataset_to_stacked would if it was asked to encode `template_dataset`.
        template_dataset: A template Dataset (or other mapping of DataArrays) demonstrating the shape of output required (variables, shapes, coordinates etc).
        preserved_dims: dimensions from the target_template that were not folded in the predictions channels. The preserved_dims need to be a subset of the dims of all the variables of template_dataset.

    Returns:
        An xarray.Dataset (or other mapping of DataArrays) with the same shape and type as template_dataset.
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
            Note even if this is set to False, the model may be able to infer the longitude from relative features, unless `relative_latitude_local_coordinates` is also True, or if there is any bias on the relative edge sizes for different longitudes.
        add_node_longitude: Add features for longitude (cos(lon), sin(lon)).
            Note even if this is set to False, the model may be able to infer the longitude from relative features, unless `relative_longitude_local_coordinates` is also True, or if there is any bias on the relative edge sizes for different longitudes.
        add_relative_positions: Whether to relative positions in R3 to the edges.
        relative_longitude_local_coordinates: If True, relative positions are computed in a local space where the receiver is at 0 longitude.
        relative_latitude_local_coordinates: If True, relative positions are computed in a local space where the receiver is at 0 latitude.
        sine_cosine_encoding: If True, we will transform the node/edge features with sine and cosine functions, similar to NERF.
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


def lat_lon_to_leading_axes(grid_xarray: "xarray.DataArray") -> "xarray.DataArray":
    """Reorders xarray so lat/lon axes come first."""
    # leading + ["lat", "lon"] + trailing
    # to
    # ["lat", "lon"] + leading + trailing
    return grid_xarray.transpose("lat", "lon", ...)


def restore_leading_axes(grid_xarray: "xarray.DataArray") -> "xarray.DataArray":
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
        latitude_local_coordinates: Whether to rotate edges such that in the positions are computed such that the receiver is always at latitude 0.
        longitude_local_coordinates: Whether to rotate edges such that in the positions are computed such that the receiver is always at longitude 0.

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
        rotate_latitude: Whether to produce a rotation matrix that would rotate R^3 vectors to zero latitude.
        rotate_longitude: Whether to produce a rotation matrix that would rotate R^3 vectors to zero longitude.

    Returns:
        Matrices of shape [leading_axis] such that when applied to the reference
            position with `rotate_with_matrices(rotation_matrices, reference_pos)`

            * phi goes to 0. if "rotate_longitude" is True.

            * theta goes to np.pi / 2 if "rotate_latitude" is True.

            The rotation consists of:
            * rotate_latitude = False, rotate_longitude = True:
                  Latitude preserving rotation.
            * rotate_latitude = True, rotate_longitude = True:
                  Latitude preserving rotation, followed by longitude preserving rotation.
            * rotate_latitude = True, rotate_longitude = False:
                  Latitude preserving rotation, followed by longitude preserving rotation, and the inverse of the latitude preserving rotation. Note this is computationally different from rotating the longitude only and is. We do it like this, so the polar geodesic curve, continues to be aligned with one of the axis after the rotation.
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
        senders_node_lat: Latitudes in the [-90, 90] interval of shape [num_sender_nodes]
        senders_node_lon: Longitudes in the [0, 360] interval of shape [num_sender_nodes]
        senders: Sender indices of shape [num_edges], indices in [0, num_sender_nodes)
        receivers_node_lat: Latitudes in the [-90, 90] interval of shape [num_receiver_nodes]
        receivers_node_lon: Longitudes in the [0, 360] interval of shape [num_receiver_nodes]
        receivers: Receiver indices of shape [num_edges], indices in [0, num_receiver_nodes)
        add_node_positions: Add unit norm absolute positions.
        add_node_latitude: Add a feature for latitude (cos(90 - lat)).
            Note even ifthis is set to False, the model may be able to infer the longitude from relative features, unless `relative_latitude_local_coordinates` is also True, or if there is any bias on the relative edge sizes for different longitudes.
        add_node_longitude: Add features for longitude (cos(lon), sin(lon)).
            Note even if this is set to False, the model may be able to infer the longitude from relative features, unless `relative_longitude_local_coordinates` is also True, or if there is any bias on the relative edge sizes for different longitudes.
        add_relative_positions: Whether to relative positions in R3 to the edges.
        edge_normalization_factor: Allows explicitly controlling edge normalization. If None, defaults to max edge length. This supports using pre-trained model weights with a different graph structure to what it was trained on.
        relative_longitude_local_coordinates: If True, relative positions are computed in a local space where the receiver is at 0 longitude.
        relative_latitude_local_coordinates: If True, relative positions are computed in a local space where the receiver is at 0 latitude.

    Returns:
          Arrays of shape: [num_nodes, num_features] and [num_edges, num_features]. with node and edge features.
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
        latitude_local_coordinates: Whether to rotate edges such that in the positions are computed such that the receiver is always at latitude 0.
        longitude_local_coordinates: Whether to rotate edges such that in the positions are computed such that the receiver is always at longitude 0.

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


class GraphGridMesh:
    """Graph datatype of GraphCast.

    Args:
        mesh_size (int): size of mesh.
        radius_query_fraction_edge_length (float): _description_
        mesh2grid_edge_normalization_factor (float): Normalization factor of edge in Mesh2Grid GNN.
        resolution (float): resolution of atmospheric data.
        mesh2mesh_src_index (np.array, optional): Index of Mesh2Mesh source node. Defaults to None.
        mesh2mesh_dst_index (np.array, optional): Index of Mesh2Mesh destination node. Defaults to None.
        grid2mesh_src_index (np.array, optional): Index of Grid2Mesh source node. Defaults to None.
        grid2mesh_dst_index (np.array, optional): Index of Grid2Mesh destination node.
        mesh2grid_src_index (np.array, optional): Index of Mesh2Grid source node. Defaults to None.
        mesh2grid_dst_index (np.array, optional): Index of Mesh2Grid destination node. Defaults to None.
        mesh_num_nodes (int, optional): Number of mesh nodes. Defaults to None.
        grid_num_nodes (int, optional): Number of grid nodes. Defaults to None.
        mesh_num_edges (int, optional): Number of mesh edges. Defaults to None.
        grid2mesh_num_edges (int, optional): Number of edges in Grid2Mesh GNN. Defaults to None.
        mesh2grid_num_edges (int, optional): Number of edges in Mesh2Grid GNN. Defaults to None.
        grid_node_feat (np.array, optional): Feature of grid nodes. Defaults to None.
        mesh_node_feat (np.array, optional): Feature of mehs nodes. Defaults to None.
        mesh_edge_feat (np.array, optional): Feature of mesh edges. Defaults to None.
        grid2mesh_edge_feat (np.array, optional): Feature of edges in Grid2Mesh GNN. Defaults to None.
        mesh2grid_edge_feat (np.array, optional): Feature of edges in Mesh2Grid GNN. Defaults to None.
    """

    def __init__(
        self,
        mesh_size: int,
        radius_query_fraction_edge_length: float,
        mesh2grid_edge_normalization_factor: float,
        resolution: float,
        mesh2mesh_src_index: np.array = None,
        mesh2mesh_dst_index: np.array = None,
        grid2mesh_src_index: np.array = None,
        grid2mesh_dst_index: np.array = None,
        mesh2grid_src_index: np.array = None,
        mesh2grid_dst_index: np.array = None,
        mesh_num_nodes: int = None,
        grid_num_nodes: int = None,
        mesh_num_edges: int = None,
        grid2mesh_num_edges: np.array = None,
        mesh2grid_num_edges: np.array = None,
        grid_node_feat: np.array = None,
        mesh_node_feat: np.array = None,
        mesh_edge_feat: np.array = None,
        grid2mesh_edge_feat: np.array = None,
        mesh2grid_edge_feat: np.array = None,
    ):
        self.meshes = get_hierarchy_of_triangular_meshes_for_sphere(mesh_size)

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
            self.query_radius = (
                self._get_max_edge_distance(self.finest_mesh)
                * radius_query_fraction_edge_length
            )
            self._mesh2grid_edge_normalization_factor = (
                mesh2grid_edge_normalization_factor
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
                grid_lat=np.arange(-90.0, 90.0 + resolution, resolution),
                grid_lon=np.arange(0.0, 360.0, resolution),
            )
            self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
            self._mesh_graph_structure = self._init_mesh_graph()
            self._mesh2grid_graph_structure = self._init_mesh2grid_graph()
        else:
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

    def tensor(self):
        self.mesh2mesh_src_index = paddle.to_tensor(
            self.mesh2mesh_src_index, dtype=paddle.int64
        )

        self.mesh2mesh_dst_index = paddle.to_tensor(
            self.mesh2mesh_dst_index, dtype=paddle.int64
        )
        self.grid2mesh_src_index = paddle.to_tensor(
            self.grid2mesh_src_index, dtype=paddle.int64
        )
        self.grid2mesh_dst_index = paddle.to_tensor(
            self.grid2mesh_dst_index, dtype=paddle.int64
        )
        self.mesh2grid_src_index = paddle.to_tensor(
            self.mesh2grid_src_index, dtype=paddle.int64
        )
        self.mesh2grid_dst_index = paddle.to_tensor(
            self.mesh2grid_dst_index, dtype=paddle.int64
        )
        self.grid_node_feat = paddle.to_tensor(
            self.grid_node_feat, dtype=paddle.get_default_dtype()
        )
        self.mesh_node_feat = paddle.to_tensor(
            self.mesh_node_feat, dtype=paddle.get_default_dtype()
        )
        self.mesh_edge_feat = paddle.to_tensor(
            self.mesh_edge_feat, dtype=paddle.get_default_dtype()
        )
        self.grid2mesh_edge_feat = paddle.to_tensor(
            self.grid2mesh_edge_feat, dtype=paddle.get_default_dtype()
        )
        self.mesh2grid_edge_feat = paddle.to_tensor(
            self.mesh2grid_edge_feat, dtype=paddle.get_default_dtype()
        )
        return self

    @property
    def finest_mesh(self):
        return self.meshes[-1]

    def init_mesh_properties(self):
        """Inits static properties that have to do with mesh nodes."""
        self.mesh_num_nodes = self.finest_mesh.vertices.shape[0]
        mesh_phi, mesh_theta = cartesian_to_spherical(
            self.finest_mesh.vertices[:, 0],
            self.finest_mesh.vertices[:, 1],
            self.finest_mesh.vertices[:, 2],
        )
        (mesh_nodes_lat, mesh_nodes_lon) = spherical_to_lat_lon(
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
        ) = get_bipartite_graph_spatial_features(
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
        node_features, edge_features = get_graph_spatial_features(
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
        (_, _, edge_features) = get_bipartite_graph_spatial_features(
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
        targets_template: "xarray.Dataset",
    ) -> "xarray.Dataset":
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
        grid_xarray = restore_leading_axes(grid_xarray_lat_lon_leading)

        # xarray `DataArray` (batch, lat, lon, channels)
        # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
        return stacked_to_dataset(grid_xarray.variable, targets_template)


class TriangularMesh(NamedTuple):
    vertices: np.ndarray
    faces: np.ndarray


def merge_meshes(mesh_list: Sequence[TriangularMesh]) -> TriangularMesh:
    for i in range(len(mesh_list) - 1):
        mesh_i, mesh_ip1 = mesh_list[i], mesh_list[i + 1]
        num_nodes_mesh_i = mesh_i.vertices.shape[0]
        assert np.allclose(mesh_i.vertices, mesh_ip1.vertices[:num_nodes_mesh_i])

    return TriangularMesh(
        vertices=mesh_list[-1].vertices,
        faces=np.concatenate([mesh.faces for mesh in mesh_list], axis=0),
    )


def get_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    product = [[1.0, phi], [1.0, -phi], [-1.0, phi], [-1.0, -phi]]
    vertices = []
    for p in product:
        c1 = p[0]
        c2 = p[1]
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
) -> List[TriangularMesh]:
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


class _ChildVerticesBuilder:
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
        faces: Integer array of shape [num_faces, 3]. Contains node indices adjacent to each face.
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for radius query.

    Args:
        grid_latitude: Latitude values for the grid [num_lat_points]
        grid_longitude: Longitude values for the grid [num_lon_points]
        mesh: Mesh object.
        radius: Radius of connectivity in R3. for a sphere of unit radius.

    Returns:
        tuple with `grid_indices` and `mesh_indices` indicating edges between the grid and the mesh such that the distances in a straight line (not geodesic) are smaller than or equal to `radius`.
        grid_indices: Indices of shape [num_edges], that index into a
        [num_lat_points, num_lon_points] grid, after flattening the leading axes.
        mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mesh-grid edge indices for grid points contained in mesh triangles.

    Args:
        grid_latitude: Latitude values for the grid [num_lat_points]
        grid_longitude: Longitude values for the grid [num_lon_points]
        mesh: Mesh object.

    Returns:
        tuple with `grid_indices` and `mesh_indices` indicating edges between the grid and the mesh vertices of the triangle that contain each grid point. The number of edges is always num_lat_points * num_lon_points * 3
        grid_indices: Indices of shape [num_edges], that index into a [num_lat_points, num_lon_points] grid, after flattening the leading axes.
        mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
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


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Computes year progress for times in seconds.
    Args:
        seconds_since_epoch: Times in seconds since the "epoch" (the point at which UNIX time starts).
    Returns:
        Year progress normalized to be in the `[0, 1)` interval for each time point.
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
        seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the point at which UNIX time starts).
        longitude: 1D array of longitudes at which day progress is computed.
    Returns:
        2D array of day progress values normalized to be in the [0, 1) inverval for each time point at each longitude.
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
    dataset: "xarray.Dataset",
    input_duration: str,
    target_lead_times: str,
):
    (target_lead_times, target_duration) = _process_target_lead_times_and_get_duration(
        target_lead_times
    )
    time = dataset.coords["time"]
    dataset = dataset.assign_coords(time=time + target_duration - time[-1])

    targets = dataset.sel({"time": target_lead_times})

    input_duration = pd.Timedelta(input_duration)
    zero = pd.Timedelta(0)
    epsilon = pd.Timedelta(1, "ns")
    inputs = dataset.sel({"time": slice(-input_duration + epsilon, zero)})
    return inputs, targets


def _process_target_lead_times_and_get_duration(target_lead_times: str):
    """Returns the minimum duration for the target lead times."""
    if isinstance(target_lead_times, slice):
        if target_lead_times.start is None:
            target_lead_times = slice(
                pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step
            )
        target_duration = pd.Timedelta(target_lead_times.stop)
    else:
        if not isinstance(target_lead_times, (list, tuple, set)):
            target_lead_times = [target_lead_times]

        target_lead_times = [pd.Timedelta(x) for x in target_lead_times]
        target_lead_times.sort()
        target_duration = target_lead_times[-1]
    return target_lead_times, target_duration


def variable_to_stacked(
    variable: "xarray.Variable",
    sizes: "xarray.core.utils.Frozen",
    preserved_dims=("batch", "lat", "lon"),
) -> "xarray.Variable":
    """Converts an xarray.Variable to preserved_dims + ("channels",).

    Any dimensions other than those included in preserved_dims get stacked into a final "channels" dimension. If any of the preserved_dims are missing then they are added, with the data broadcast/tiled to match the sizes specified in `sizes`.

    Args:
        variable: An xarray.Variable.
        sizes: Mapping including sizes for any dimensions which are not present in `variable` but are needed for the output. This may be needed for example for a static variable with only ("lat", "lon") dims, or if you want to encode just the latitude coordinates (a variable with dims ("lat",)).
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
    dataset: "xarray.Dataset",
    sizes=None,
    preserved_dims=("batch", "lat", "lon"),
) -> "xarray.DataArray":
    """Converts an xarray.Dataset to a single stacked array.

    This takes each consistuent data_var, converts it into BHWC layout
    using `variable_to_stacked`, then concats them all along the channels axis.

    Args:
        dataset: An xarray.Dataset.
        sizes: Mapping including sizes for any dimensions which are not present in the `dataset` but are needed for the output. See variable_to_stacked.
        preserved_dims: dimensions from the dataset that should not be folded in the predictions channels.

    Returns:
        An xarray.DataArray with dimensions preserved_dims + ("channels",). Existing coordinates for preserved_dims axes will be preserved, however there will be no coordinates for "channels".
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


class GridMeshAtmosphericDataset(io.Dataset):
    """This class is used to process ERA5 re-analyze data, and is used to generate the dataset generator supported by MindSpore. This class inherits the Data class.

    Args:
        input_keys (Tuple[str, ...]): Name of input data.
        label_keys (Tuple[str, ...]): Name of label data.
        data_path: Path of atmospheric datafile.
        mean_path: Path of mean datafile.
        stddev_path: Path of standard deviation datafile.
        stddev_diffs_path: Path of standard deviation different datafile.
        type: Type of GraphCast network.
        mesh_size: Size of mesh.
        mesh2grid_edge_normalization_factor: Factor of normalization of edges in Mesh2Grid GNN.
        radius_query_fraction_edge_length: Length of radius query fraction edges.
        resolution: Resolution of atmospheric data.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.GridMeshAtmosphericDataset(
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "data_path": "/path/to/file.nc",
        ...     "mean_path": "/path/to/file.nc",
        ...     "stddev_path": "/path/to/file.nc",
        ...     "stddev_diffs_path": "/path/to/file.nc",
        ...     "type": "graphcast_small",
        ...     "mesh_size": 5,
        ...     "mesh2grid_edge_normalization_factor": 0.06,
        ...     "radius_query_fraction_edge_length": 0.6180338738074472,
        ...     "resolution": 1,
        ... )  # doctest: +SKIP
    """

    use_graph_grid_mesh: bool = True

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        data_path: str,
        mean_path: str,
        stddev_path: str,
        stddev_diffs_path: str,
        type: str,
        mesh_size: int,
        mesh2grid_edge_normalization_factor: float,
        radius_query_fraction_edge_length: float,
        resolution: float,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        if type == "graphcast":
            self.input_variables = TASK_input_variables
            self.forcing_variables = TASK_forcing_variables
            self.target_variables = TASK_target_variables
            self.level_variables = PRESSURE_LEVELS[37]
        elif type == "graphcast_small":
            self.input_variables = TASK_13_input_variables
            self.forcing_variables = TASK_13_forcing_variables
            self.target_variables = TASK_13_target_variables
            self.level_variables = PRESSURE_LEVELS[13]
        elif type == "graphcast_operational":
            self.input_variables = TASK_13_PRECIP_OUT_input_variables
            self.forcing_variables = TASK_13_PRECIP_OUT_forcing_variables
            self.target_variables = TASK_13_PRECIP_OUT_target_variables
            self.level_variables = PRESSURE_LEVELS[13]

        nc_dataset = xarray.open_dataset(data_path)

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

        stddev_data = xarray.open_dataset(stddev_path).sel(
            level=list(self.level_variables)
        )
        stddev_diffs_data = xarray.open_dataset(stddev_diffs_path).sel(
            level=list(self.level_variables)
        )
        mean_data = xarray.open_dataset(mean_path).sel(level=list(self.level_variables))

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

        lat_dim, lon_dim, batch_dim, feat_dim = stacked_inputs.shape
        stacked_inputs = stacked_inputs.data.reshape(lat_dim * lon_dim, batch_dim, -1)
        stacked_targets = stacked_targets.data.reshape(lat_dim * lon_dim, batch_dim, -1)
        self.stacked_targets_stddev = stacked_targets_stddev.data
        self.stacked_targets_mean = stacked_targets_mean.data.reshape(
            lat_dim * lon_dim, batch_dim, -1
        )

        self.input_data = []
        self.target_data = []

        graph = GraphGridMesh(
            mesh_size=mesh_size,
            radius_query_fraction_edge_length=radius_query_fraction_edge_length,
            mesh2grid_edge_normalization_factor=mesh2grid_edge_normalization_factor,
            resolution=resolution,
        )

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
        return (
            {
                self.input_keys[0]: self.input_data[idx],
            },
            {
                self.label_keys[0]: self.target_data[idx],
            },
            None,
        )

    def normalize(self, inputs_data, stddev_data, mean_data):
        for name in list(inputs_data.keys()):
            inputs_data[name] = (inputs_data[name] - mean_data[name]) / stddev_data[
                name
            ]
        return inputs_data

    def denormalize(self, inputs_data):
        return inputs_data * self.stacked_targets_stddev + self.stacked_targets_mean

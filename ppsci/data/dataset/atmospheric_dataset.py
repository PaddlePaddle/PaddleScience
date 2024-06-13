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

from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from paddle import io

import ppsci.data.dataset.atmospheric_utils as atmospheric_utils

try:
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
    variable: "xarray.Variable",
    sizes,
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
    """
    This class is used to process ERA5 re-analyze data,
    and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        input_keys (Tuple[str, ...]): Name of input data.
        label_keys (Tuple[str, ...]): Name of label data.
        config: Configuration of graph.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.GridMeshAtmosphericDataset(
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ...     "config": config,
        ... )  # doctest: +SKIP

    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        config: Dict[str, Union[int, float, str]],
    ):
        super().__init__()
        self.input_keys = input_keys
        self.label_keys = label_keys
        if config.type == "graphcast":
            self.input_variables = TASK_input_variables
            self.forcing_variables = TASK_forcing_variables
            self.target_variables = TASK_target_variables
            self.level_variables = PRESSURE_LEVELS[37]
        elif config.type == "graphcast_small":
            self.input_variables = TASK_13_input_variables
            self.forcing_variables = TASK_13_forcing_variables
            self.target_variables = TASK_13_target_variables
            self.level_variables = PRESSURE_LEVELS[13]
        elif config.type == "graphcast_operational":
            self.input_variables = TASK_13_PRECIP_OUT_input_variables
            self.forcing_variables = TASK_13_PRECIP_OUT_forcing_variables
            self.target_variables = TASK_13_PRECIP_OUT_target_variables
            self.level_variables = PRESSURE_LEVELS[13]

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

        graph = atmospheric_utils.GraphGridMesh(config)

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

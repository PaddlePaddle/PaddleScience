import copy
import os
import pickle
import typing

import args
import graphtype
import numpy as np
import paddle
import pandas as pd
import xarray

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


class ERA5Data(paddle.io.Dataset):
    """
    This class is used to process ERA5 re-analyze data,
    and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training,
        evaluation or testing. Supports [“train”,“test”, “valid”].
        Default: 'train'.

    Supported Platforms:
        ``Ascend`` ``GPU``

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
        # TODO:处理完整数据集进行训练，处理过程同本函数处理过程
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
            graph_template = graphtype.GraphGridMesh(config)

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

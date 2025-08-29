from datetime import datetime
from typing import Tuple

import xarray as xr


def date_to_hours(date: str):
    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
    return hours_since_jan_01_epoch


def get_mean_std(mean_path: str, std_path: str, vars_channel: Tuple[int, ...] = None):
    data_mean = xr.open_mfdataset(mean_path)["mean"].values
    data_std = xr.open_mfdataset(std_path)["std"].values

    data_mean.resize(data_mean.shape[0], 1, 1)
    data_std.resize(data_std.shape[0], 1, 1)

    return data_mean, data_std

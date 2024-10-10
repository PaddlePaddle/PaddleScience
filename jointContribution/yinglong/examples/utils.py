from datetime import datetime
from typing import Optional
from typing import Tuple

import numpy as np


def date_to_hours(date: str):
    date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
    return hours_since_jan_01_epoch


def get_mean_std(mean_path: str, std_path: str, vars_channel: Tuple[int, ...]):
    data_mean = np.load(mean_path).reshape(-1, 1, 1).astype(np.float32)
    data_mean = data_mean[vars_channel]
    data_std = np.load(std_path).reshape(-1, 1, 1).astype(np.float32)
    data_std = data_std[vars_channel]
    return data_mean, data_std


def get_time_mean(
    time_mean_path: str,
    img_h: int,
    img_w: int,
    vars_channel: Optional[Tuple[int, ...]] = None,
):
    time_mean = np.load(time_mean_path).astype(np.float32)
    if vars_channel is not None:
        time_mean = time_mean[vars_channel, :img_h, :img_w]
    else:
        time_mean = time_mean[:, :img_h, :img_w]
    return time_mean

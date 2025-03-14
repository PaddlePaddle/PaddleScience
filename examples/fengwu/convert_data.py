# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ref: https://github.com/HaxyMoly/Pangu-Weather-ReadyToGo/blob/main/forecast_decode_functions.py

import os
from os import path as osp
from typing import Dict

import hydra
import netCDF4 as nc
import numpy as np

from ppsci.utils import logger


def convert_surface_data_to_nc(
    surface_file: str, file_name: str, output_dir: str
) -> None:
    surface_data = np.load(surface_file)
    u_component_of_wind_10m = surface_data[0]
    v_component_of_wind_10m = surface_data[1]
    temperature_2m = surface_data[2]
    mean_sea_level_pressure = surface_data[3]

    with nc.Dataset(
        os.path.join(output_dir, file_name), "w", format="NETCDF4_CLASSIC"
    ) as nc_file:
        # Create dimensions
        nc_file.createDimension("longitude", 1440)
        nc_file.createDimension("latitude", 721)

        # Create variables
        nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude",))
        nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude",))
        nc_msl = nc_file.createVariable(
            "mean_sea_level_pressure", np.float32, ("latitude", "longitude")
        )
        nc_u10 = nc_file.createVariable(
            "u_component_of_wind_10m", np.float32, ("latitude", "longitude")
        )
        nc_v10 = nc_file.createVariable(
            "v_component_of_wind_10m", np.float32, ("latitude", "longitude")
        )
        nc_t2m = nc_file.createVariable(
            "temperature_2m", np.float32, ("latitude", "longitude")
        )

        # Set variable attributes
        nc_lon.units = "degrees_east"
        nc_lat.units = "degrees_north"
        nc_msl.units = "Pa"
        nc_u10.units = "m/s"
        nc_v10.units = "m/s"
        nc_t2m.units = "K"

        # Write data to variables
        nc_lon[:] = np.linspace(0.125, 359.875, 1440)
        nc_lat[:] = np.linspace(90, -90, 721)
        nc_msl[:] = mean_sea_level_pressure
        nc_u10[:] = u_component_of_wind_10m
        nc_v10[:] = v_component_of_wind_10m
        nc_t2m[:] = temperature_2m

    logger.info(
        f"Convert output surface data file {surface_file} as nc format and save to {output_dir}/{file_name}."
    )


def convert_upper_data_to_nc(upper_file: str, file_name: str, output_dir: str) -> None:
    # Load the saved numpy arrays
    upper_data = np.load(upper_file)

    # surface data offset
    st = 4
    level = 13

    geopotential = upper_data[st : st + level]
    specific_humidity = upper_data[st + level : st + 2 * level]
    u_component_of_wind = upper_data[st + 2 * level : st + 3 * level]
    v_component_of_wind = upper_data[st + 3 * level : st + 4 * level]
    temperature = upper_data[st + 4 * level :]

    with nc.Dataset(
        os.path.join(output_dir, file_name), "w", format="NETCDF4_CLASSIC"
    ) as nc_file:
        # Create dimensions
        nc_file.createDimension("longitude", 1440)
        nc_file.createDimension("latitude", 721)
        nc_file.createDimension("level", level)

        # Create variables
        nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude",))
        nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude",))
        nc_geopotential = nc_file.createVariable(
            "geopotential", np.float32, ("level", "latitude", "longitude")
        )
        nc_specific_humidity = nc_file.createVariable(
            "specific_humidity", np.float32, ("level", "latitude", "longitude")
        )
        nc_temperature = nc_file.createVariable(
            "temperature", np.float32, ("level", "latitude", "longitude")
        )
        nc_u_component_of_wind = nc_file.createVariable(
            "u_component_of_wind", np.float32, ("level", "latitude", "longitude")
        )
        nc_v_component_of_wind = nc_file.createVariable(
            "v_component_of_wind", np.float32, ("level", "latitude", "longitude")
        )

        # Set variable attributes
        nc_lon.units = "degrees_east"
        nc_lat.units = "degrees_north"
        nc_geopotential.units = "m"
        nc_specific_humidity.units = "kg/kg"
        nc_temperature.units = "K"
        nc_u_component_of_wind.units = "m/s"
        nc_v_component_of_wind.units = "m/s"
        # Write data to variables
        nc_lon[:] = np.linspace(0.125, 359.875, 1440)
        nc_lat[:] = np.linspace(90, -90, 721)
        nc_geopotential[:] = geopotential
        nc_specific_humidity[:] = specific_humidity
        nc_temperature[:] = temperature
        nc_u_component_of_wind[:] = u_component_of_wind
        nc_v_component_of_wind[:] = v_component_of_wind

    logger.info(
        f"Convert output upper data file {upper_file} as nc format and save to {output_dir}/{file_name}."
    )


def convert(cfg: Dict):
    output_dir = cfg.output_dir

    for _, file_name in os.listdir(output_dir):
        if not file_name.endwiths("npy"):
            continue

        convert_surface_data_to_nc(
            osp.join(output_dir, file_name),
            osp.basename(file_name) + "_surface.nc",
            output_dir,
        )
        convert_upper_data_to_nc(
            osp.join(output_dir, file_name),
            osp.basename(file_name) + "_upper.nc",
            output_dir,
        )


@hydra.main(version_base=None, config_path="./conf", config_name="fengwu.yaml")
def main(cfg: Dict):
    if cfg.mode == "infer":
        convert(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['infer'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

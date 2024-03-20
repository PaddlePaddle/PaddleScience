from dataclasses import dataclass
from dataclasses import field

import numpy as np

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

# The list of all possible atmospheric variables. Taken from:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Table9
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

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


@dataclass
class TrainingArguments:

    data_path: str = field(
        default="data/dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc",
        metadata={"help": "data_path."},
    )
    param_path: str = field(
        default="data/params/GraphCast---ERA5-1979-2017---resolution-0.25---pressure-levels-37---mesh-2to6---precipitation-input-and-output.pdparams",
        metadata={"help": "param_path."},
    )
    stddev_path: str = field(
        default="data/stats/stddev_by_level.nc",
        metadata={"help": "stddev_path."},
    )
    stddev_diffs_path: str = field(
        default="data/stats/diffs_stddev_by_level.nc",
        metadata={"help": "stddev_diffs_path."},
    )
    mean_path: str = field(
        default="data/stats/mean_by_level.nc",
        metadata={"help": "mean_path."},
    )
    type: str = field(
        default="graphcast",
        metadata={"help": "type."},
    )
    level: int = field(
        default=37,
        metadata={"help": "level."},
    )
    latent_size: int = field(
        default=512,
        metadata={"help": "latent_size."},
    )
    hidden_layers: int = field(
        default=1,
        metadata={"help": "hidden_layers."},
    )
    gnn_msg_steps: int = field(
        default=16,
        metadata={"help": "gnn_msg_steps."},
    )
    mesh_size: int = field(
        default=6,
        metadata={"help": "mesh_size."},
    )
    resolution: float = field(
        default=0.25,
        metadata={"help": "resolution. {0.25, 1.0}"},
    )
    radius_query_fraction_edge_length: float = field(
        default=0.6,
        metadata={"help": "radius_query_fraction_edge_length."},
    )
    mesh2grid_edge_normalization_factor: float = field(
        default=2 / (1 + np.sqrt(5)),
        metadata={"help": "mesh2grid_edge_normalization_factor. 1 / phi"},
    )

    # 输入数据
    mesh_node_dim: int = field(
        default=474,
        metadata={"help": "mesh_node_dim."},
    )
    grid_node_dim: int = field(
        default=474,
        metadata={"help": "grid_node_dim."},
    )
    mesh_edge_dim: int = field(
        default=4,
        metadata={"help": "mesh_edge_dim."},
    )
    grid2mesh_edge_dim: int = field(
        default=4,
        metadata={"help": "grid2mesh_edge_dim."},
    )
    mesh2grid_edge_dim: int = field(
        default=4,
        metadata={"help": "mesh2grid_edge_dim."},
    )

    # 测试数据
    mesh_node_num: int = field(
        default=2562,
        metadata={"help": "mesh_node_num."},
    )
    grid_node_num: int = field(
        default=32768,
        metadata={"help": "grid_node_num."},
    )
    mesh_edge_num: int = field(
        default=20460,
        metadata={"help": "mesh_edge_num."},
    )
    mesh2grid_edge_num: int = field(
        default=98304,
        metadata={"help": "mesh2grid_edge_num."},
    )
    grid2mesh_edge_num: int = field(
        default=50184,
        metadata={"help": "grid2mesh_edge_num."},
    )
    # 输出结果
    node_output_dim: int = field(
        default=227,
        metadata={"help": "node_output_dim."},
    )

    @property
    def grid_node_emb_dim(self):
        return self.latent_size

    @property
    def mesh_node_emb_dim(self):
        return self.latent_size

    @property
    def mesh_edge_emb_dim(self):
        return self.latent_size

    @property
    def grid2mesh_edge_emb_dim(self):
        return self.latent_size

    @property
    def mesh2grid_edge_emb_dim(self):
        return self.latent_size


if __name__ == "__main__":
    import json

    with open("GraphCast_small.json", "r") as f:
        args = TrainingArguments(**json.load(f))
    print(args)

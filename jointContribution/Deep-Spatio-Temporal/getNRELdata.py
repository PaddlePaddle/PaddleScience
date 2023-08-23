#!/usr/bin/env python
"""
    Get wind speed data from NREL WIND
    https://www.nrel.gov/grid/wind-toolkit.html
    Select one wind farm with 100 turbines from Wyoming
"""
import h5pyd
import numpy as np
import pandas as pd

f = h5pyd.File("/nrel/wtk/conus/wtk_conus_2012.h5", "r")
meta = pd.DataFrame(f["meta"][()])

lon = -105.243988
lat = 41.868515
df = meta[
    (meta["longitude"] < lon + 0.25)
    & (meta["longitude"] >= lon)
    & (meta["latitude"] <= lat + 0.03)
    & (meta["latitude"] > lat - 0.18)
]

df = df.drop(
    [
        864121,
        868456,
        869542,
        870629,
        871718,
        872807,
        873897,
        876088,
        866300,
        867383,
        868467,
        869553,
        870640,
    ]
)
df.to_csv("./data/wind_speed_meta.csv")
gid_list = list(df.index)
wind_speed_list = []
for gid in gid_list:
    wind_speed_list.append(f["windspeed_100m"][:, gid])

time_array = f["time_index"][()]
wind_speed_array = np.vstack(wind_speed_list)

wind_speed_df = pd.DataFrame(wind_speed_array, index=df.index, columns=time_array)
wind_speed_df = wind_speed_df / f["windspeed_100m"].attrs["scale_factor"]

wind_speed_df.to_csv("./data/wind_speed.csv")

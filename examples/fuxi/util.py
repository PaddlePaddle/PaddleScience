import os

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["save_like"]

pl_names = ["z", "t", "u", "v", "r"]
sfc_names = ["t2m", "u10", "v10", "msl", "tp"]
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def weighted_rmse(out, tgt):
    wlat = np.cos(np.deg2rad(tgt.lat))
    wlat /= wlat.mean()
    error = (out - tgt) ** 2 * wlat
    return np.sqrt(error.mean(("lat", "lon")))


def split_variable(ds, name):
    if name in sfc_names:
        v = ds.sel(level=[name])
        v = v.assign_coords(level=[0])
        v = v.rename({"level": "level0"})
        v = v.transpose("member", "level0", "time", "dtime", "lat", "lon")
    elif name in pl_names:
        level = [f"{name}{l}" for l in levels]
        v = ds.sel(level=level)
        v = v.assign_coords(level=levels)
        v = v.transpose("member", "level", "time", "dtime", "lat", "lon")
    return v


def save_like(output, input, step, save_dir="", freq=6, split=False):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        step = (step + 1) * freq
        init_time = pd.to_datetime(input.time.values[-1])

        ds = xr.DataArray(
            output[None],
            dims=["time", "step", "level", "lat", "lon"],
            coords=dict(
                time=[init_time],
                step=[step],
                level=input.level,
                lat=input.lat.values,
                lon=input.lon.values,
            ),
        ).astype(np.float32)

        if split:

            def rename(name):
                if name == "tp":
                    return "TP06"
                elif name == "r":
                    return "RH"
                return name.upper()

            new_ds = []
            for k in pl_names + sfc_names:
                v = split_variable(ds, k)
                v.name = rename(k)
                new_ds.append(v)
            ds = xr.merge(new_ds, compat="no_conflicts")

        save_name = os.path.join(save_dir, f"{step:03d}.nc")
        # print(f'Save to {save_name} ...')
        ds.to_netcdf(save_name)


def test_rmse(output_name, target_name):
    output = xr.open_dataarray(output_name)
    output = output.isel(time=0).sel(step=120)
    target = xr.open_dataarray(target_name)

    for level in ["z500", "t850", "t2m", "u10", "v10", "msl", "tp"]:
        out = output.sel(level=level)
        tgt = target.sel(level=level)
        rmse = weighted_rmse(out, tgt).load()
        print(f"{level.upper()} 120h rmse: {rmse:.3f}")

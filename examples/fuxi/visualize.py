import argparse
import os

import numpy as np
import xarray as xr


def visualize(save_name, vars=[], titles=[], vmin=None, vmax=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig_height = 4 if len(vars) == 1 else 4 * len(vars)
    fig, ax = plt.subplots(
        len(vars),
        1,
        figsize=(7, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    plt.subplots_adjust(hspace=0.25)

    def plot(ax, v, title):
        im = v.plot(
            ax=ax,
            x="lon",
            y="lat",
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False,
            cmap="viridis",
        )

        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.03,
            aspect=20,
            shrink=0.6,
            fraction=0.04,
            anchor=(0.0, 0.5),
        )
        cbar.set_label(
            v.name if hasattr(v, "name") else "Value", fontsize=9, labelpad=2
        )
        cbar.ax.tick_params(labelsize=7)

        # ax.coastlines()
        ax.set_title(title, fontsize=12)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

    for i, v in enumerate(vars):
        if len(vars) == 1:
            plot(ax, v, titles[i])
        else:
            plot(ax[i], v, titles[i])

    plt.savefig(
        save_name, bbox_inches="tight", pad_inches=0.1, transparent="true", dpi=200
    )
    plt.close()


def test_visualize(step, data_dir, save_dir):
    src_name = os.path.join(data_dir, f"{step:03d}.nc")
    ds = xr.open_dataarray(src_name).isel(time=0)
    ds = ds.sel(lon=slice(0, 360), lat=slice(90, -90))
    print(ds)
    u850 = ds.sel(level="U850", step=step)
    v850 = ds.sel(level="V850", step=step)
    ws850 = np.sqrt(u850**2 + v850**2)
    visualize(
        os.path.join(save_dir, f"{step:03d}.jpg"),
        [ws850],
        [f"850 hPa Wind Speed Forecasting (m/s) in 20231012-00+{step:03d}h"],
        vmin=0,
        vmax=30,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="The input data dir"
    )
    parser.add_argument("--save_dir", type=str, default="output_fuxi")
    parser.add_argument("--step", type=int, required=True, help="the predict step")
    args = parser.parse_args()

    test_visualize(args.step, args.data_dir, args.save_dir)

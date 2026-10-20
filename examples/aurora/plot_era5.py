"""Plot full-grid ERA5 truth, framework predictions, and Paddle forecast error."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from evaluate_era5 import align_grid


def main():
    plt.switch_backend("Agg")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--torch", type=Path, required=True)
    p.add_argument("--paddle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for variable, level, label, offset, scale in (
        ("surf_2t", None, "2 m temperature (°C)", 273.15, 1.0),
        ("atmos_z", 500, "500 hPa geopotential height (m)", 0.0, 9.80665),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)
        for step, row in enumerate(axes, 1):
            with xr.open_dataset(
                args.paddle / f"prediction_{step:03}.nc"
            ) as pd, xr.open_dataset(
                args.torch / f"prediction_{step:03}.nc"
            ) as pt, xr.open_dataset(
                args.data / f"truth_{step:03}.nc"
            ) as raw:
                truth = align_grid(raw, pd)
                pt = align_grid(pt, pd)

                def values(ds):
                    field = ds[variable].isel(batch=0, history=0)
                    if level is not None:
                        field = field.sel(level=level)
                    return (field.values - offset) / scale

                target, reference, actual = values(truth), values(pt), values(pd)
                error = actual - target
                low, high = np.percentile(target, [1, 99])
                bound = max(float(np.percentile(np.abs(error), 99)), 1e-6)
                extent = [
                    float(pd.longitude[0]),
                    float(pd.longitude[-1]),
                    float(pd.latitude[-1]),
                    float(pd.latitude[0]),
                ]
                for ax, data, title, cmap, vmin, vmax in (
                    (row[0], target, "ERA5", "coolwarm", low, high),
                    (row[1], reference, "Torch", "coolwarm", low, high),
                    (row[2], actual, "Paddle", "coolwarm", low, high),
                    (row[3], error, "Paddle - ERA5", "RdBu_r", -bound, bound),
                ):
                    im = ax.imshow(
                        data,
                        extent=extent,
                        origin="upper",
                        aspect="auto",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_title(f"{title}, +{step*6}h")
                    ax.set_xlabel("Longitude (°E)")
                    ax.set_ylabel("Latitude (°N)")
                    fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(f"AuroraPretrained: {label}; 2023-01-01 06 UTC initialization")
        fig.tight_layout()
        path = args.output / ("temperature.png" if level is None else "z500.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()

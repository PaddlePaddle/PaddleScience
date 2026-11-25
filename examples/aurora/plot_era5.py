"""Plot ERA5 truth, Paddle forecasts, and physical forecast errors."""

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
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.data / "input.nc") as initial:
        initialization = initial.time.values[0]
    for variable, level, label, offset, scale in (
        ("surf_2t", None, "2 m temperature (°C)", 273.15, 1.0),
        ("atmos_z", 500, "500 hPa geopotential height (m)", 0.0, 9.80665),
    ):
        fig, axes = plt.subplots(
            args.steps, 3, figsize=(15, 3.5 * args.steps), squeeze=False
        )
        for step, row in enumerate(axes, 1):
            with xr.open_dataset(
                args.prediction / f"prediction_{step:03}.nc"
            ) as pd, xr.open_dataset(args.data / f"truth_{step:03}.nc") as raw:
                truth = align_grid(raw, pd)

                def values(ds):
                    field = ds[variable].isel(batch=0, history=0)
                    if level is not None:
                        field = field.sel(level=level)
                    return (field.values - offset) / scale

                target, actual = values(truth), values(pd)
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
                    (row[1], actual, "Paddle", "coolwarm", low, high),
                    (row[2], error, "Paddle - ERA5", "RdBu_r", -bound, bound),
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
        fig.suptitle(
            f"AuroraPretrained: {label}; initialized {str(initialization)[:16]} UTC"
        )
        fig.tight_layout()
        path = args.output / ("temperature.png" if level is None else "z500.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()

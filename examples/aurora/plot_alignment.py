"""Plot Torch/Paddle temperatures and their differences for a saved rollout."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    fig, axes = plt.subplots(
        args.steps, 3, figsize=(15, 3.7 * args.steps), squeeze=False
    )
    for step, row in enumerate(axes, 1):
        with xr.open_dataset(
            args.directory / f"torch-prediction-{step:03}.nc"
        ) as ref, xr.open_dataset(
            args.directory / f"paddle-prediction-{step:03}.nc"
        ) as pred:
            left = ref.surf_2t.values[0, 0] - 273.15
            right = pred.surf_2t.values[0, 0] - 273.15
            error = right - left
            low, high = np.percentile(left, [1, 99])
            bound = max(float(np.abs(error).max()), 1e-6)
            time = str(ref.time.values[0])[:16]
            for ax, value, title, cmap, vmin, vmax in (
                (row[0], left, "Torch", "coolwarm", low, high),
                (row[1], right, "Paddle", "coolwarm", low, high),
                (row[2], error, "Paddle - Torch", "RdBu_r", -bound, bound),
            ):
                im = ax.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_title(f"{title}: {time} UTC")
                ax.set_xlabel("Longitude index")
                ax.set_ylabel("Latitude index")
                fig.colorbar(im, ax=ax, label="2 m temperature (°C)", shrink=0.8)
    fig.suptitle("Aurora official sample: framework regression (not forecast skill)")
    fig.tight_layout()
    path = args.directory / "temperature-comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()

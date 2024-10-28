r"""Interactive test and visualization of vector flow derivatives."""

from typing import Dict
from typing import Optional
from typing import Sequence

import deepali.core.bspline as B
import deepali.core.functional as U
import matplotlib.pyplot as plt
import paddle
from deepali.core import Axes
from deepali.core import Grid
from deepali.utils import paddle_aux  # noqa

# %%
# Imports

# %%
# Auxiliary functions
def change_axes(flow: paddle.Tensor, grid: Grid, axes: Axes, to_axes: Axes) -> paddle.Tensor:
    if axes != to_axes:
        flow = U.move_dim(flow, 1, -1)
        flow = grid.transform_vectors(flow, axes=axes, to_axes=to_axes)
        flow = U.move_dim(flow, -1, 1)
    return flow


def flow_derivatives(
    flow: paddle.Tensor, grid: Grid, axes: Axes, to_axes: Optional[Axes] = None, **kwargs
) -> Dict[str, paddle.Tensor]:
    if to_axes is None:
        to_axes = axes
    flow = change_axes(flow, grid, axes, to_axes)
    axes = to_axes
    if "spacing" not in kwargs:
        if axes == Axes.CUBE:
            spacing = tuple(2 / n for n in tuple(grid.shape))
        elif axes == Axes.CUBE_CORNERS:
            spacing = tuple(2 / (n - 1) for n in tuple(grid.shape))
        elif axes == Axes.GRID:
            spacing = 1
        elif axes == Axes.WORLD:
            spacing = grid.spacing()
        else:
            spacing = None
        kwargs["spacing"] = spacing
    return U.flow_derivatives(flow, **kwargs)


def random_svf(size: Sequence[int], stride: int = 1, generator=None) -> paddle.Tensor:
    cp_grid_size = B.cubic_bspline_control_point_grid_size(size, stride=stride)
    cp_grid_size = tuple(reversed(cp_grid_size))
    data = paddle.randn(shape=(1, 3) + cp_grid_size)
    data = U.fill_border(data, margin=3, value=0, inplace=True)
    return B.evaluate_cubic_bspline(data, size=size, stride=stride)


def visualize_flow(
    ax: plt.Axes,
    flow: paddle.Tensor,
    grid: Optional[Grid] = None,
    axes: Optional[Axes] = None,
    label: Optional[str] = None,
) -> None:
    if grid is None:
        grid = Grid(shape=tuple(flow.shape)[2:])
    if axes is None:
        axes = grid.axes()
    flow = change_axes(flow, grid, axes, grid.axes())
    x = grid.coords(channels_last=False, dtype=flow.dtype, device=flow.place)
    x = U.move_dim(x.unsqueeze_(axis=0).add_(y=paddle.to_tensor(flow)), 1, -1)
    target_grid = U.grid_image(shape=tuple(flow.shape)[2:], inverted=True, stride=(5, 5))
    warped_grid = U.warp_image(target_grid, x, align_corners=grid.align_corners())
    ax.imshow(warped_grid[0, 0, tuple(flow.shape)[2] // 2], cmap="gray")
    if label:
        ax.set_title(label, fontsize=24)


# %%
# Random velocity fields
generator = paddle.framework.core.default_cpu_generator().manual_seed(42)
grid = Grid(size=(128, 128, 64), spacing=(0.5, 0.5, 1.0))
flow = random_svf(tuple(grid.shape), stride=8, generator=generator).multiply_(
    y=paddle.to_tensor(0.1)
)
fig, axes = plt.subplots(1, 1, figsize=(4, 4))
ax = axes
ax.set_title("v", fontsize=24, pad=20)
visualize_flow(ax, flow, grid=grid, axes=grid.axes())


# %%
# Visualise first order derivatives for different modes
configs = [
    dict(mode="forward_central_backward"),
    dict(mode="bspline"),
    dict(mode="gaussian", sigma=0.7355),
]

fig, axes = plt.subplots(len(configs), 4, figsize=(16, 4 * len(configs)))

for i, config in enumerate(configs):
    derivs = flow_derivatives(
        flow,
        grid=grid,
        axes=grid.axes(),
        to_axes=Axes.GRID,
        which=["du/dx", "du/dy", "dv/dx", "dv/dy"],
        **config,
    )
    for ax, (key, deriv) in zip(axes[i], derivs.items()):
        if i == 0:
            ax.set_title(key, fontsize=24, pad=20)
        ax.imshow(deriv[0, 0, tuple(deriv.shape)[2] // 2], vmin=-1, vmax=1)


# %%
# Compare magnitudes of first order derivatives for different modes
flow_axes = [Axes.GRID, Axes.WORLD, Axes.CUBE_CORNERS]
sigma = 0.7355
configs = [
    dict(mode="bspline"),
    dict(mode="gaussian", sigma=sigma),
    dict(mode="forward_central_backward", sigma=sigma),
    dict(mode="forward_central_backward"),
]

for to_axes in flow_axes:
    for config in configs:
        print(f"axes={to_axes}, " + ", ".join(f"{k}={v!r}" for k, v in config.items()))
        derivs = flow_derivatives(
            flow,
            grid=grid,
            axes=grid.axes(),
            to_axes=to_axes,
            which=["du/dx", "du/dy", "dv/dx", "dv/dy"],
            **config,
        )
        for key, deriv in derivs.items():
            print(f"- max(abs({key})): {deriv.abs().max().item():.5f}")
        print()
    print("\n")


# %%
# Visualise second order derivatives for different modes
configs = [
    dict(mode="forward_central_backward"),
    dict(mode="bspline"),
    dict(mode="gaussian", sigma=0.7355),
]

fig, axes = plt.subplots(len(configs), 4, figsize=(16, 4 * len(configs)))

for i, config in enumerate(configs):
    derivs = flow_derivatives(
        flow,
        grid=grid,
        axes=grid.axes(),
        to_axes=Axes.GRID,
        which=["du/dxx", "du/dxy", "dv/dxy", "dv/dyy"],
        **config,
    )
    for ax, (key, deriv) in zip(axes[i], derivs.items()):
        if i == 0:
            ax.set_title(key, fontsize=24, pad=20)
        ax.imshow(deriv[0, 0, tuple(deriv.shape)[2] // 2], vmin=-0.4, vmax=0.4)


# %%
# Compare magnitudes of second order derivatives for different modes
flow_axes = [Axes.GRID, Axes.WORLD, Axes.CUBE_CORNERS]

sigma = 0.7355

configs = [
    dict(mode="bspline"),
    dict(mode="gaussian", sigma=sigma),
    dict(mode="forward_central_backward", sigma=sigma),
    dict(mode="forward_central_backward"),
]

for to_axes in flow_axes:
    for config in configs:
        print(f"axes={to_axes}, " + ", ".join(f"{k}={v!r}" for k, v in config.items()))
        derivs = flow_derivatives(
            flow,
            grid=grid,
            axes=grid.axes(),
            to_axes=to_axes,
            which=["du/dxx", "du/dxy", "dv/dxy", "dv/dyy"],
            **config,
        )
        for key, deriv in derivs.items():
            print(f"- max(abs({key})): {deriv.abs().max().item():.5f}")
        print()
    print("\n")

# %%

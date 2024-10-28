# %%
# Imports
from typing import Sequence

import deepali.core.bspline as B
import deepali.core.functional as U
import deepali.utils.paddle_aux  # noqa
import matplotlib.pyplot as plt
import paddle
from deepali.core import Grid


# %%
# Auxiliary functions
def random_svf(size: Sequence[int], stride: int = 1, generator=None) -> paddle.Tensor:
    cp_grid_size = B.cubic_bspline_control_point_grid_size(size, stride=stride)
    data = paddle.randn(shape=(1, 3) + cp_grid_size)
    data = U.fill_border(data, margin=3, value=0, inplace=True)
    return B.evaluate_cubic_bspline(data, size=size, stride=stride)


def visualize_flow(ax, flow: paddle.Tensor) -> None:
    grid = Grid(shape=tuple(flow.shape)[2:], align_corners=True)
    x = grid.coords(channels_last=False, dtype=u.dtype, device=u.place)
    x = U.move_dim(x.unsqueeze(axis=0).add_(y=paddle.to_tensor(flow)), 1, -1)
    target_grid = U.grid_image(shape=tuple(flow.shape)[2:], inverted=True, stride=(5, 5))
    warped_grid = U.warp_image(target_grid, x)
    ax.imshow(warped_grid[0, 0, tuple(flow.shape)[2] // 2], cmap="gray")


# %%
# Random velocity fields
size = 128, 128, 128
generator = paddle.framework.core.default_cpu_generator().manual_seed(42)
u = random_svf(size, stride=8, generator=generator).multiply_(y=paddle.to_tensor(0.1))
v = random_svf(size, stride=8, generator=generator).multiply_(y=paddle.to_tensor(0.05))


# %%
# Evaluate displacement fields
flow_u = U.expv(u)
flow_v = U.expv(v)
flow = U.compose_flows(flow_u, flow_v)


# %%
# Approximate velocity field of composite displacement field
flow_w = U.expv(U.compose_svfs(u, v, bch_terms=3))


# %%
# Visualize composite displacement fields and error norm
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

visualize_flow(axes[0], flow)
visualize_flow(axes[1], flow_w)

error = flow_w.sub(flow).norm(axis=1, keepdim=True)

ax = axes[2]
_ = ax.imshow(error[0, 0, tuple(error.shape)[2] // 2], cmap="jet", vmin=0, vmax=0.1)


# %%

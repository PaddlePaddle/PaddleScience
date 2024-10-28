# %%
import math
from timeit import default_timer as timer
from typing import Callable
from typing import Tuple

import paddle
from deepali.core import bspline as B
from deepali.core import functional as U
from deepali.core.enum import SpatialDim
from deepali.utils import paddle_aux  # noqa

# %% Vector field control point coefficients
# device = torch.device("cuda:0")
device = paddle.CPUPlace()
in_size = tuple((21,))  # (X, ...)
stride = (5,) * len(in_size)
cp_size = B.cubic_bspline_control_point_grid_size(in_size, stride)
cp_data = paddle.arange(dtype="float32", end=len(in_size) * cp_size.size)
cp_data = cp_data.reshape(1, len(cp_size), *tuple(reversed(cp_size)))


# %% Reference implementation based on MIRTK C++ code


def compute_bspline_indices_and_weights_1d(
    x: float, degree: int = 3
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if degree & 1:
        i = int(math.floor(x)) - degree // 2
    else:
        i = int(math.floor(x + 0.5)) - degree // 2
    i = paddle.arange(start=i, end=i + degree + 1, step=1, dtype="int32")
    wx = paddle.empty(shape=[4], dtype="float32")
    if degree == 3:
        w = x - i[1]
        wx[3] = 1 / 6 * w * w * w
        wx[0] = 1 / 6 + 1 / 2 * w * (w - 1) - wx[3]
        wx[2] = w + wx[0] - 2 * wx[3]
        wx[1] = 1 - wx[0] - wx[2] - wx[3]
    else:
        raise NotImplementedError(f"compute_bspline_indices_and_weights_1d() for degree={degree}")
    return i, wx


def compute_bspline_indices_and_weights_2d(
    x: float, y: float, degree: int = 3
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if degree & 1:
        i = int(math.floor(x)) - degree // 2
        j = int(math.floor(y)) - degree // 2
    else:
        i = int(math.floor(x + 0.5)) - degree // 2
        j = int(math.floor(y + 0.5)) - degree // 2
    i = paddle.arange(start=i, end=i + degree + 1, step=1, dtype="int32")
    j = paddle.arange(start=j, end=j + degree + 1, step=1, dtype="int32")
    wx = paddle.empty(shape=[4], dtype="float32")
    wy = paddle.empty(shape=[4], dtype="float32")
    if degree == 3:
        w = x - i[1]
        wx[3] = 1 / 6 * w * w * w
        wx[0] = 1 / 6 + 1 / 2 * w * (w - 1) - wx[3]
        wx[2] = w + wx[0] - 2 * wx[3]
        wx[1] = 1 - wx[0] - wx[2] - wx[3]
        w = y - j[1]
        wy[3] = 1 / 6 * w * w * w
        wy[0] = 1 / 6 + 1 / 2 * w * (w - 1) - wy[3]
        wy[2] = w + wy[0] - 2 * wy[3]
        wy[1] = 1 - wy[0] - wy[2] - wy[3]
    else:
        raise NotImplementedError(f"compute_bspline_indices_and_weights_2d() for degree={degree}")
    return i, j, wx, wy


def interpolate_cubic_bspline_1d(data: paddle.Tensor, x: float) -> paddle.Tensor:
    degree = 3
    i, w = compute_bspline_indices_and_weights_1d(x, degree=degree)
    w = w.to(data)
    val = paddle.zeros(shape=tuple(data.shape)[:2]).to(data)
    for a in range(degree + 1):
        ia: int = max(0, min(i[a].item(), tuple(data.shape)[2] - 1))
        val += data[:, :, ia].mul(w[a])
    return val


def interpolate_cubic_bspline_2d(data: paddle.Tensor, x: float, y: float) -> paddle.Tensor:
    degree = 3
    i, j, wx, wy = compute_bspline_indices_and_weights_2d(x, y, degree=degree)
    wx = wx.to(data)
    wy = wy.to(data)
    val = paddle.zeros(shape=tuple(data.shape)[:2]).to(data)
    for b in range(degree + 1):
        jb: int = max(0, min(j[b].item(), tuple(data.shape)[2] - 1))
        for a in range(degree + 1):
            ia: int = max(0, min(i[a].item(), tuple(data.shape)[3] - 1))
            val += data[..., jb, ia].mul(wx[a] * wy[b])
    return val


# %% Evaluate B-spline values at output points
D = cp_data.ndim - 2
N = tuple(cp_data.shape)[0]
C = tuple(cp_data.shape)[1]
conv_fn: Callable[..., paddle.Tensor] = [
    paddle.nn.functional.conv1d,
    paddle.nn.functional.conv2d,
    paddle.nn.functional.conv3d,
][D - 1]
kernels = B.bspline_interpolation_weights(
    degree=3, stride=stride, dtype=cp_data.dtype, device=cp_data.place
)
start = timer()
output = cp_data
for dim, kernel in zip((SpatialDim(dim).tensor_dim(cp_data.ndim) for dim in range(D)), kernels):
    weight = kernel.reshape((tuple(kernel.shape)[0], 1, tuple(kernel.shape)[1]) + (1,) * (D - 1))
    weight = weight.tile(repeat_times=(C,) + (1,) * (weight.ndim - 1))
    output = U.move_dim(output, dim, 2)
    output = conv_fn(output, weight, groups=C)
    output = output.reshape((N, C, tuple(kernel.shape)[0]) + tuple(output.shape)[2:])
    output = output.transpose(perm=paddle_aux.transpose_aux_func(output.ndim, 2, 3)).flatten(
        start_axis=2, stop_axis=3
    )
    print(output)
    output = U.move_dim(output, 2, dim)
output = output[(slice(0, N), slice(0, C)) + tuple(slice(0, n) for n in reversed(in_size))]
output = output.contiguous()
print(f"Elapsed time: {timer() - start:.3f}s")
assert tuple(output.shape)[0] == N
assert tuple(output.shape)[1] == C
assert tuple(output.shape)[2:] == tuple(reversed(in_size))


# %%
kernel = tuple(B.cubic_bspline1d(s) for s in stride)
for _ in range(3):
    start = timer()
    result1 = B.evaluate_cubic_bspline(
        cp_data, size=in_size, stride=stride, kernel=kernel, transpose=True
    )
    print(f"Elapsed time: {timer() - start:.3f}s")
kernel = B.bspline_interpolation_weights(
    degree=3, stride=stride, dtype=cp_data.dtype, device=cp_data.place
)
for _ in range(3):
    start = timer()
    result2 = B.evaluate_cubic_bspline(
        cp_data, size=in_size, stride=stride, kernel=kernel, transpose=False
    )
    print(f"Elapsed time: {timer() - start:.3f}s")

assert paddle.allclose(x=result1, y=result2, atol=0.01).item()


# %% Cubic B-spline kernel and its derivatives
import matplotlib.pyplot as plt  # noqa

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

stride = 7

cp_data = paddle.zeros(shape=(1, 1, 11))
cp_data[0, 0, (tuple(cp_data.shape)[2] - 1) // 2] = 1

for derivative in range(3):
    kernel = B.cubic_bspline_interpolation_weights(stride=stride, derivative=derivative)
    values = B.evaluate_cubic_bspline(cp_data, stride=stride, kernel=kernel)
    ax.plot(values[0, 0])


# %%

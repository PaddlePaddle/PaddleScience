"""ide Volterra integro-differential equation (examples/ide/volterra_ide.py)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "ide"
NOTE = (
    "ppsci.equation.Volterra builds its integral matrix with numpy inside forward, so the "
    "Gauss-Legendre points and the [N, N+N*Q] kernel*weight matrix are precomputed in "
    "build_batch; residual u'+u - int_mat@u as in the example; Adam lr 1e-3 instead of LBFGS"
)
BOUND, UPPER = 0.0, 5.0
N, Q = 12, 20  # TRAIN.npoint_interior, TRAIN.quad_deg
BATCH_SIZE = N + N * Q + 1  # interior + quad points + 1 IC point


def build_model():
    return ppsci.arch.MLP(
        input_keys=("x",),
        output_keys=("u",),
        num_layers=3,
        hidden_size=20,
        activation="tanh",
    )


def build_batch():
    x = np.sort(np.random.uniform(BOUND, UPPER, [N, 1]), axis=0).astype("float32")
    qx, qw = np.polynomial.legendre.leggauss(Q)
    x_quad = (x - BOUND) / 2 * qx[None, :] + (x + BOUND) / 2  # [N, Q]
    int_mat = np.zeros([N, N + N * Q], "float32")
    for i in range(N):
        kernel = np.exp(x_quad[i] - x[i, 0])  # kernel_func(x, s) = exp(s - x)
        int_mat[i, N + Q * i : N + Q * (i + 1)] = (x[i, 0] - BOUND) / 2 * qw * kernel
    xs = paddle.to_tensor(
        np.concatenate([x, x_quad.reshape([-1, 1])]).astype("float32")
    )
    xs.stop_gradient = False
    inputs = {"x": xs, "ic_x": paddle.zeros([1, 1])}
    labels = {
        "int_mat": paddle.to_tensor(int_mat),
        "u": paddle.ones([1, 1]),
    }  # exp(-0)cosh(0)
    return inputs, labels


def forward(model, inputs):
    return model({"x": inputs["x"]})


def compute_loss(model, inputs, labels):
    x = inputs["x"]
    u = model({"x": x})["u"]
    lhs = ppsci.autodiff.jacobian(u, x) + u
    volterra = lhs[:N] - paddle.mm(labels["int_mat"], u)
    ppsci.autodiff.clear()
    loss = C.sq_mean(volterra)
    return loss + C.mse(model({"x": inputs["ic_x"]})["u"], labels["u"])

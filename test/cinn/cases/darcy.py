"""darcy 2D Darcy flow as a Poisson problem (examples/darcy/darcy2d.py)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "darcy"
NOTE = (
    "uniform random interior/boundary points instead of evenly-spaced geometry samples; "
    "constant lr 1e-3 instead of OneCycleLR"
)
NPOINT_PDE = 99**2
NPOINT_BC = 400
BATCH_SIZE = NPOINT_PDE + NPOINT_BC
KEYS = ("x", "y")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("p",),
        num_layers=5,
        hidden_size=20,
        activation="stan",
    )


def _ref(x, y):
    return paddle.sin(2.0 * np.pi * x) * paddle.cos(2.0 * np.pi * y)


def build_batch():
    inputs = C.points(KEYS, NPOINT_PDE)
    # 100 random points on each side of the unit square
    s = C.uniform([NPOINT_BC, 1])
    side = paddle.arange(NPOINT_BC).reshape([-1, 1]) // (NPOINT_BC // 4)
    zeros, ones = paddle.zeros_like(s), paddle.ones_like(s)
    bx = paddle.where(side == 0, zeros, paddle.where(side == 1, ones, s))
    by = paddle.where(side == 2, zeros, paddle.where(side == 3, ones, s))
    inputs.update({"bc_x": bx, "bc_y": by})
    labels = {
        "poisson": -8.0 * np.pi**2 * _ref(inputs["x"], inputs["y"]).detach(),
        "bc_p": _ref(bx, by),
    }
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    EQUATIONS.update(C.lambdify(ppsci.equation.Poisson(2).equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    data = C.take(inputs, KEYS)
    out = model(data)
    res = EQUATIONS["poisson"]({**data, **out})
    C.autodiff.clear()
    loss = paddle.sum(paddle.square(res - labels["poisson"]))  # MSELoss("sum")
    bc_out = model(C.take(inputs, KEYS, "bc_"))
    loss += paddle.sum(paddle.square(bc_out["p"] - labels["bc_p"]))
    return loss

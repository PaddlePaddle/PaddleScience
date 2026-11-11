"""nsfnet VP-NSFNet1, Kovasznay flow Re=40 (examples/nsfnet/VP_NSFNet1.py)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "nsfnet"
NOTE = (
    "interior points from paddle.uniform in the example's box; boundary points/labels "
    "built as in generate_data; plain Adam (no piecewise lr, no LBFGS stage)"
)
N_TRAIN = 2601
NB_TRAIN = 400
BATCH_SIZE = N_TRAIN + NB_TRAIN
RE = 40
LAM = 0.5 * RE - np.sqrt(0.25 * RE**2 + 4 * np.pi**2)
KEYS = ("x", "y")


def build_model():
    return ppsci.arch.MLP(KEYS, ("u", "v", "p"), 4, 50, "tanh")


def build_batch():
    inputs = {  # interior: x in [-0.5, 1.0], y in [-0.5, 1.5]
        "x": C.uniform([N_TRAIN, 1], -0.5, 1.0, requires_grad=True),
        "y": C.uniform([N_TRAIN, 1], -0.5, 1.5, requires_grad=True),
    }
    x = np.linspace(-0.5, 1.0, 101)
    y = np.linspace(-0.5, 1.5, 101)
    yb = np.concatenate([y[1:101], y[0:100], [-0.5] * 100, [1.5] * 100])
    xb = np.concatenate([[-0.5] * 100, [1] * 100, x[0:100], x[1:101]])
    xb = xb.reshape(-1, 1).astype("float32")
    yb = yb.reshape(-1, 1).astype("float32")
    ub = 1 - np.exp(LAM * xb) * np.cos(2 * np.pi * yb)
    vb = LAM / (2 * np.pi) * np.exp(LAM * xb) * np.sin(2 * np.pi * yb)
    inputs["bc_x"] = paddle.to_tensor(xb)
    inputs["bc_y"] = paddle.to_tensor(yb)
    labels = {"u": paddle.to_tensor(ub), "v": paddle.to_tensor(vb)}
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    ns = ppsci.equation.NavierStokes(1.0 / RE, 1.0, 2, False)
    EQUATIONS.update(C.lambdify(ns.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.data_loss(model(C.take(inputs, KEYS, "bc_")), labels)
    loss += C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    return loss

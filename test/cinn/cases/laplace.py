"""laplace 2D Laplace equation (examples/laplace/laplace2d.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "laplace"
NOTE = "interior points uniform in [0,1]^2, boundary points random on the 4 edges, instead of geometry sampling"
NPOINT_INTERIOR = 9801
NPOINT_BC = 400
NPOINT_TOTAL = NPOINT_INTERIOR + NPOINT_BC  # EQ batch size in the example
BATCH_SIZE = NPOINT_TOTAL + NPOINT_BC
KEYS = ("x", "y")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS, output_keys=("u",), num_layers=5, hidden_size=20
    )


def build_batch():
    inputs = C.points(KEYS, NPOINT_TOTAL, 0.0, 1.0)
    s = paddle.uniform([NPOINT_BC // 4, 1], min=0.0, max=1.0)
    zero, one = paddle.zeros_like(s), paddle.ones_like(s)
    bx = paddle.concat([s, s, zero, one])
    by = paddle.concat([zero, one, s, s])
    inputs.update({"bc_x": bx, "bc_y": by})
    return inputs, {"u": paddle.cos(bx) * paddle.cosh(by)}


EQUATIONS = {}


def prepare(model):
    EQUATIONS.update(C.lambdify(ppsci.equation.Laplace(dim=2).equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    # both constraints use MSELoss("sum")
    out = model(C.take(inputs, KEYS))
    res = EQUATIONS["laplace"]({**C.take(inputs, KEYS), **out})
    ppsci.autodiff.clear()
    loss = paddle.sum(paddle.square(res))
    out = model(C.take(inputs, KEYS, "bc_"))
    return loss + paddle.sum(paddle.square(out["u"] - labels["u"]))

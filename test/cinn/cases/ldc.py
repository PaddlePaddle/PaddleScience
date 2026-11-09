"""ldc 2D steady lid-driven cavity, Re=10 (examples/ldc/ldc2d_steady_Re10.py)."""
from cases import _common as C

import ppsci

EXAMPLE = "ldc"
NOTE = "collocation points sampled uniformly in [-0.05, 0.05]^2 instead of from the geometry"
NPOINT_PDE = 99**2
NPOINT_BC = 101
BATCH_SIZE = NPOINT_PDE + 4 * NPOINT_BC
KEYS = ("x", "y")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u", "v", "p"),
        num_layers=9,
        hidden_size=50,
        activation="tanh",
    )


def build_batch():
    inputs = C.points(KEYS, NPOINT_PDE, -0.05, 0.05)
    for side in ("top", "bottom", "left", "right"):
        inputs.update(C.points(KEYS, NPOINT_BC, -0.05, 0.05, prefix=f"{side}_"))
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    ns = ppsci.equation.NavierStokes(0.01, 1.0, 2, False)
    EQUATIONS.update(C.lambdify(ns.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    loss += C.bc_loss(model, C.take(inputs, KEYS, "top_"), {"u": 1.0, "v": 0.0})
    for side in ("bottom", "left", "right"):
        loss += C.bc_loss(model, C.take(inputs, KEYS, f"{side}_"), {"u": 0.0, "v": 0.0})
    return loss

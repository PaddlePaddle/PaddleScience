"""heat_pinn 2D steady heat conduction, Laplace (examples/heat_pinn/heat_pinn.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "heat_pinn"
NOTE = "collocation/boundary points sampled uniformly instead of from the Rectangle geometry"
LEARNING_RATE = 5e-4
NPOINT_PDE = 99**2
NPOINT_BC = 25
BATCH_SIZE = NPOINT_PDE + 4 * NPOINT_BC
KEYS = ("x", "y")
# side -> (fixed coord, value, u target)
SIDES = {
    "top": ("y", 1.0, 0.0),
    "bottom": ("y", -1.0, 50 / 75),
    "left": ("x", -1.0, 1.0),
    "right": ("x", 1.0, 0.0),
}


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u",),
        num_layers=9,
        hidden_size=20,
        activation="tanh",
    )


def build_batch():
    inputs = C.points(KEYS, NPOINT_PDE, -1.0, 1.0)
    for side, (key, value, _) in SIDES.items():
        pts = C.points(KEYS, NPOINT_BC, -1.0, 1.0, prefix=f"{side}_")
        pts[f"{side}_{key}"] = paddle.full([NPOINT_BC, 1], value)
        inputs.update(pts)
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    EQUATIONS.update(C.lambdify(ppsci.equation.Laplace(dim=2).equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    for side, (_, _, target) in SIDES.items():
        loss += 0.25 * C.bc_loss(model, C.take(inputs, KEYS, f"{side}_"), {"u": target})
    return loss

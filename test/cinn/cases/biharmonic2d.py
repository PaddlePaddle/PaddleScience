"""biharmonic2d Kirchhoff plate bending (examples/biharmonic2d/biharmonic2d.py)."""
import numpy as np
import paddle
import sympy as sp
from cases import _common as C

import ppsci

EXAMPLE = "biharmonic2d"
NOTE = "random collocation points on the plate / its edges instead of geometry sampling; Adam only (no LBFGS stage)"
E, NU, Q_0, LENGTH, WIDTH, HEIGHT = 201880.0e6, 0.25, 980, 2, 3, 0.01
N_BC, N_PDE = 125, 8000
BATCH_SIZE = N_PDE + 8 * N_BC
KEYS = ("x", "y")
W_BC, W_PDE = 100.0, 1.0
# side -> (fixed key, fixed value, free key, free upper bound)
SIDES = {
    "left": ("x", 0.0, "y", WIDTH),
    "right": ("x", float(LENGTH), "y", WIDTH),
    "up": ("y", 0.0, "x", LENGTH),
    "bottom": ("y", float(WIDTH), "x", LENGTH),
}


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u",),
        num_layers=5,
        hidden_size=20,
        activation="tanh",
        weight_norm=True,
    )


def _edge(side, prefix):
    fixed, value, free, high = SIDES[side]
    pts = {
        free: C.uniform([N_BC, 1], 0.0, high, requires_grad=True),
        fixed: paddle.full([N_BC, 1], value, dtype="float32"),
    }
    pts[fixed].stop_gradient = False
    return {f"{prefix}{side}_{k}": v for k, v in pts.items()}


def build_batch():
    inputs = {
        "x": C.uniform([N_PDE, 1], 0.0, LENGTH, requires_grad=True),
        "y": C.uniform([N_PDE, 1], 0.0, WIDTH, requires_grad=True),
    }
    for side in SIDES:
        inputs.update(_edge(side, "w_"))
        inputs.update(_edge(side, "m_"))
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    x, y = sp.symbols("x y")
    q = Q_0 * sp.sin(np.pi * x / LENGTH) * sp.sin(np.pi * y / WIDTH)
    D = E * (HEIGHT**3) / (12.0 * (1.0 - NU**2))
    pde = ppsci.equation.Biharmonic(dim=2, q=q, D=D)
    EQUATIONS.update(C.lambdify(pde.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    hessian = C.autodiff.hessian
    loss = W_PDE * C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    for side in SIDES:
        out = model(C.take(inputs, KEYS, f"w_{side}_"))
        loss = loss + W_BC * C.sq_mean(out["u"])
    for side in SIDES:
        d = C.take(inputs, KEYS, f"m_{side}_")
        u = model(d)["u"]
        if side in ("left", "right"):  # M_y
            m = -(NU * hessian(u, d["x"]) + hessian(u, d["y"]))
        else:  # M_x
            m = -(hessian(u, d["x"]) + NU * hessian(u, d["y"]))
        loss = loss + W_BC * C.sq_mean(m)
    C.autodiff.clear()
    return loss

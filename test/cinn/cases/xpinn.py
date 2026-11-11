"""xpinn 2D Poisson with 3 subdomains (examples/xpinn/xpinn.py)."""
import os
import sys

from cases import _common as C

_EX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "xpinn")
sys.path.insert(0, os.path.abspath(_EX_DIR))
import model as xpinn_model  # noqa: E402
from xpinn import loss_fun  # noqa: E402  examples/xpinn/xpinn.py

# importing xpinn.py also calls set_prim_eager_enabled(True), as the example does

EXAMPLE = "xpinn"
NOTE = "points uniform in [0, 1]^2 and random boundary labels instead of XPINN_2D_PoissonEqn.mat"
N_B, N_F1, N_F2, N_F3, N_I1, N_I2 = 200, 5000, 1800, 1200, 100, 100
BATCH_SIZE = N_B + N_F1 + N_F2 + N_F3 + N_I1 + N_I2
LEARNING_RATE = 8e-4
LAYERS = ([2, 30, 30, 1], [2, 20, 20, 20, 20, 1], [2, 25, 25, 25, 1])


def build_model():
    return xpinn_model.Model(LAYERS)


def build_batch():
    inputs = {}
    for name, n in (
        ("residual1", N_F1),
        ("residual2", N_F2),
        ("residual3", N_F3),
        ("interface1", N_I1),
        ("interface2", N_I2),
        ("boundary", N_B),
    ):
        inputs.update(C.points(("x", "y"), n, 0.0, 1.0, prefix=f"{name}_"))
    return inputs, {"boundary_u_exact": C.uniform([N_B, 1], 0.0, 2.0)}


def compute_loss(model, inputs, labels):
    return loss_fun(model(inputs), labels)["residuals"]

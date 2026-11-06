"""ntopo 2D Beam2D topology optimization, displacement-net step (examples/ntopo, ntopo_2d.yaml)."""
import os
import sys

import numpy as np
import paddle
import paddle.nn.functional as F
from cases import _common as C

import ppsci

_EX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "ntopo")
sys.path.insert(0, os.path.abspath(_EX_DIR))
import equation as ntopo_equation  # noqa: E402
import model as ntopo_model  # noqa: E402

EXAMPLE = "ntopo"
NOTE = (
    "trains the INTERIOR_DISP constraint (E_xy energy with detached densities + point "
    "force) with plain Adam (yaml uses beta2=0.99, eps=1e-7); points uniform in the beam "
    "instead of geometry sampling; density/OC (MMSE, Gaussian filter) loop not covered"
)
BATCH_SIZE = 7500
LEARNING_RATE = 3e-4
NU, E, VOLUME_RATIO, ALPHA, EXPONENT = 0.3, 1.0, 0.5, 5.0, 3.0
GEO_ORIGIN, GEO_DIM = (0.0, 0.0), (1.5, 0.5)
VOLUME = float(np.prod(GEO_DIM))
FORCE = -0.0025
LAMBDA_ = NU * E / (1 - NU * NU)
MU = E / (1 + NU)
IN_KEYS = ("x_scaled", "y_scaled", "sin_x_scaled", "sin_y_scaled")


def transform_in(_in):
    x, y = _in["x"], _in["y"]
    x_scaled = 2.0 / GEO_DIM[0] * x + (-1.0 - 2.0 * GEO_ORIGIN[0] / GEO_DIM[0])
    y_scaled = 2.0 / GEO_DIM[1] * y + (-1.0 - 2.0 * GEO_ORIGIN[1] / GEO_DIM[1])
    return {
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "sin_x_scaled": paddle.sin(x_scaled),
        "sin_y_scaled": paddle.sin(y_scaled),
    }


def transform_out_disp(_in, _out):
    x = GEO_DIM[0] / 2 * (1 + _in["x_scaled"]) + GEO_ORIGIN[0]
    return {"u": x * _out["u"], "v": x * _out["v"]}


def transform_out_density(_in, _out):
    offset = np.log(VOLUME_RATIO / (1.0 - VOLUME_RATIO))
    return {"densities": F.sigmoid(ALPHA * _out["density"] + offset)}


def build_model():
    disp = ntopo_model.DenseSIRENModel(IN_KEYS, ("u", "v"), 6, 60, 1.0e-3)
    density = ntopo_model.DenseSIRENModel(IN_KEYS, ("density",), 6, 60, 1.0e-3)
    disp.register_input_transform(transform_in)
    disp.register_output_transform(transform_out_disp)
    density.register_input_transform(transform_in)
    density.register_output_transform(transform_out_density)
    return ppsci.arch.ModelList((disp, density))


def build_batch():
    inputs = {
        "x": C.uniform([BATCH_SIZE, 1], GEO_ORIGIN[0], GEO_DIM[0], requires_grad=True),
        "y": C.uniform([BATCH_SIZE, 1], GEO_ORIGIN[1], GEO_DIM[1], requires_grad=True),
        "force_x": paddle.to_tensor([[1.5]], dtype="float32"),
        "force_y": paddle.to_tensor([[0.0]], dtype="float32"),
    }
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    pde = ntopo_equation.EEquation({"lambda_": LAMBDA_, "mu": MU}, dim=2)
    EQUATIONS.update(pde.equations)


def forward(model, inputs):
    return model(C.take(inputs, ("x", "y")))


def compute_loss(model, inputs, labels):
    xy = C.take(inputs, ("x", "y"))
    out = {**xy, **model(xy)}
    E_xy = EQUATIONS["E_xy"](out)
    densities = out["densities"].detach().clone()
    loss_E = VOLUME * paddle.mean(paddle.pow(densities, EXPONENT) * E_xy, keepdim=True)
    v = model.model_list[0](C.take(inputs, ("x", "y"), "force_"))["v"]
    loss_force = -paddle.mean(v * FORCE, keepdim=True)
    ppsci.autodiff.clear()
    return (loss_E + loss_force).sum()

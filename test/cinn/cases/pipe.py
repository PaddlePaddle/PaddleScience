"""pipe Poiseuille flow with parametric nu (examples/pipe/poiseuille_flow.py)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "pipe"
NOTE = "(x, y, nu) sampled uniformly in the example's ranges instead of the shuffled meshgrid"
BATCH_SIZE = 128
LEARNING_RATE = 5e-3
NU_MEAN, NU_STD = 0.001, 0.9
L, R, RHO, P_OUT, P_IN, X_IN = 1.0, 0.05, 1, 0, 0.1, 0
X_OUT = X_IN + L
IN_KEYS = ("sin(x)", "cos(x)", "y", "nu")


def input_trans(input):
    x, y, nu = input["x"], input["y"], input["nu"]
    b = 2 * np.pi / (X_OUT - X_IN)
    c = np.pi * (X_IN + X_OUT) / (X_IN - X_OUT)
    sin_x = X_IN * paddle.sin(b * x + c)
    cos_x = X_IN * paddle.cos(b * x + c)
    return {"sin(x)": sin_x, "cos(x)": cos_x, "x": x, "y": y, "nu": nu}


def output_trans_u(input, out):
    return {"u": out["u"] * (R**2 - input["y"] ** 2)}


def output_trans_v(input, out):
    return {"v": (R**2 - input["y"] ** 2) * out["v"]}


def output_trans_p(input, out):
    return {
        "p": (P_IN - P_OUT) * (X_OUT - input["x"]) / L
        + (X_IN - input["x"]) * (X_OUT - input["x"]) * out["p"]
    }


def build_model():
    nets = []
    for key, trans in (
        ("u", output_trans_u),
        ("v", output_trans_v),
        ("p", output_trans_p),
    ):
        net = ppsci.arch.MLP(IN_KEYS, (key,), 3, 50, "swish")
        net.register_input_transform(input_trans)
        net.register_output_transform(trans)
        nets.append(net)
    return ppsci.arch.ModelList(tuple(nets))


def build_batch():
    nu_lo, nu_hi = NU_MEAN - NU_MEAN * NU_STD, NU_MEAN + NU_MEAN * NU_STD
    inputs = {
        "x": C.uniform([BATCH_SIZE, 1], X_IN, X_OUT, requires_grad=True),
        "y": C.uniform([BATCH_SIZE, 1], -R, R, requires_grad=True),
        "nu": C.uniform([BATCH_SIZE, 1], nu_lo, nu_hi, requires_grad=True),
    }
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    ns = ppsci.equation.NavierStokes(nu="nu", rho=RHO, dim=2, time=False)
    EQUATIONS.update(C.lambdify(ns.equations, model))


def compute_loss(model, inputs, labels):
    return C.pde_loss(model, inputs, EQUATIONS)

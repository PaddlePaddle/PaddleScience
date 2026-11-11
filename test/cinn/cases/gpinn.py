"""gpinn 1D gradient-enhanced PINN Poisson (examples/gpinn/poisson_1d.py)."""
import numpy as np
import paddle
import sympy as sp
from cases import _common as C

import ppsci

EXAMPLE = "gpinn"
NOTE = "uniform random points in (0, pi) instead of evenly spaced; gPINN1D PDE copied locally"
BATCH_SIZE = 15


class gPINN1D(ppsci.equation.PDE):
    def __init__(self, invar, outvar):
        super().__init__()
        x = self.create_symbols(invar)
        u = self.create_function(outvar, (x,))
        f = 8 * sp.sin(8 * x)
        for i in range(1, 5):
            f += i * sp.sin(i * x)
        df_x = (
            sp.cos(x)
            + 4 * sp.cos(2 * x)
            + 9 * sp.cos(3 * x)
            + 16 * sp.cos(4 * x)
            + 64 * sp.cos(8 * x)
        )
        self.add_equation("res1", -u.diff(x, 2) - f)
        self.add_equation("res2", -u.diff(x, 3) - df_x)


def build_model():
    model = ppsci.arch.MLP(("x",), ("u",), 3, 20, "tanh")

    def output_transform(in_, out):
        x = in_["x"]
        return {"u": x + paddle.tanh(x) * paddle.tanh(np.pi - x) * out["u"]}

    model.register_output_transform(output_transform)
    return model


def build_batch():
    return C.points(("x",), BATCH_SIZE, 0.0, np.pi), {}


EQUATIONS = {}


def prepare(model):
    EQUATIONS.update(C.lambdify(gPINN1D("x", "u").equations, model))


def compute_loss(model, inputs, labels):
    return C.pde_loss(model, inputs, EQUATIONS, weights={"res2": 0.01})

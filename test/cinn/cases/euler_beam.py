"""euler_beam 1D Euler-Bernoulli beam (examples/euler_beam/euler_beam.py)."""
import paddle
from cases import _common as C

import ppsci
from ppsci.autodiff import hessian, jacobian

EXAMPLE = "euler_beam"
NOTE = "uniform random interior points instead of Hammersley"
NPOINT_PDE = 100
NPOINT_BC = 4
BATCH_SIZE = NPOINT_PDE + NPOINT_BC


def build_model():
    return ppsci.arch.MLP(("x",), ("u",), 3, 20)


def build_batch():
    inputs = C.points(("x",), NPOINT_PDE)
    bc_x = paddle.to_tensor([[0.0], [0.0], [1.0], [1.0]])  # evenly sampled boundary
    bc_x.stop_gradient = False
    inputs["bc_x"] = bc_x
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    eq = ppsci.equation.Biharmonic(dim=1, q=-1.0, D=1.0)
    EQUATIONS.update(C.lambdify(eq.equations, model))


def forward(model, inputs):
    return model({"x": inputs["x"]})


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(model, {"x": inputs["x"]}, EQUATIONS)
    x = inputs["bc_x"]
    u = model({"x": x})["u"]
    u_xx = hessian(u, x)
    bc = (
        paddle.sum(paddle.square(u[0:1]))
        + paddle.sum(paddle.square(jacobian(u, x)[1:2]))
        + paddle.sum(paddle.square(u_xx[2:3]))
        + paddle.sum(paddle.square(jacobian(u_xx, x)[3:4]))
    )  # MSELoss("sum"), all targets 0
    C.autodiff.clear()
    return loss + bc

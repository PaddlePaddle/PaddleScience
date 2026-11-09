"""allen_cahn 1D Allen-Cahn, plain MLP (examples/allen_cahn/allen_cahn_plain.py, conf/allen_cahn.yaml)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "allen_cahn"
NOTE = (
    "plain variant (allen_cahn.yaml); random (t, x) in [0,1]x[-1,1] instead of the "
    "ContinuousNamedArrayDataset; IC u0 = x^2 cos(pi x) on 512 points instead of allen_cahn.mat; "
    "constant lr instead of ExponentialDecay"
)
NPOINT_PDE = 4096
NPOINT_IC = 512
BATCH_SIZE = NPOINT_PDE + NPOINT_IC
KEYS = ("t", "x")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u",),
        num_layers=4,
        hidden_size=256,
        activation="tanh",
        periods={"x": (2.0, False)},
    )


def build_batch():
    inputs = {
        "t": C.uniform([NPOINT_PDE, 1], 0.0, 1.0, requires_grad=True),
        "x": C.uniform([NPOINT_PDE, 1], -1.0, 1.0, requires_grad=True),
    }
    x0 = paddle.linspace(-1.0, 1.0, NPOINT_IC, dtype="float32").reshape([-1, 1])
    inputs["ic_t"] = paddle.zeros([NPOINT_IC, 1], dtype="float32")
    inputs["ic_x"] = x0
    labels = {"u0": x0**2 * paddle.cos(np.pi * x0)}
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    # AllenCahn defines its residual as a python callable, which ppsci.solver uses as-is
    EQUATIONS.update(ppsci.equation.AllenCahn(eps=0.01).equations)


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    out = model(C.take(inputs, KEYS, "ic_"))
    return loss + C.mse(out["u"], labels["u0"])

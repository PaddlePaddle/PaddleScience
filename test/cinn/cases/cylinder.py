"""cylinder 2D unsteady flow around a cylinder, Re=100
(examples/cylinder/2d_unsteady/cylinder2d_unsteady_Re100.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "cylinder"
NOTE = (
    "random points in t∈[1,50], x∈[-8,25], y∈[-8,8] and random labels instead of the CSV "
    "datasets; IC/probe sizes guessed (9662 IC points at t=1, 150 probe samples)"
)
KEYS = ("t", "x", "y")
NTIME = 30  # TRAIN_NUM_TIMESTAMPS
NPOINT_PDE = 9420 * NTIME
NPOINT_INLET = 161 * NTIME
NPOINT_OUTLET = 81 * NTIME
NPOINT_IC = 9420 + 161 + 81
NPOINT_SUP = 150
BATCH_SIZE = NPOINT_PDE + NPOINT_INLET + NPOINT_OUTLET + NPOINT_IC + NPOINT_SUP
# constraint name -> (point count, {label key: weight})
SUP = {
    "inlet": (NPOINT_INLET, {"u": 10.0, "v": 10.0}),
    "outlet": (NPOINT_OUTLET, {"p": 1.0}),
    "ic": (NPOINT_IC, {"u": 10.0, "v": 10.0, "p": 10.0}),
    "sup": (NPOINT_SUP, {"u": 10.0, "v": 10.0}),
}


def _txy(n, prefix="", requires_grad=False):
    return {
        prefix + "t": C.uniform([n, 1], 1.0, 50.0, requires_grad),
        prefix + "x": C.uniform([n, 1], -8.0, 25.0, requires_grad),
        prefix + "y": C.uniform([n, 1], -8.0, 8.0, requires_grad),
    }


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("u", "v", "p"),
        num_layers=5,
        hidden_size=50,
        activation="tanh",
    )


def build_batch():
    inputs = _txy(NPOINT_PDE, requires_grad=True)
    labels = {}
    ranges = {"u": (0.0, 1.5), "v": (-0.5, 0.5), "p": (-0.5, 0.5)}
    for name, (n, weights) in SUP.items():
        inputs.update(_txy(n, prefix=f"{name}_"))
        for k in weights:
            labels[f"{name}_{k}"] = C.uniform([n, 1], *ranges[k])
    inputs["ic_t"] = paddle.full([NPOINT_IC, 1], 1.0)  # IC at t0 = TIME_START
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    ns = ppsci.equation.NavierStokes(0.02, 1.0, 2, True)
    EQUATIONS.update(C.lambdify(ns.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    for name, (_, weights) in SUP.items():
        out = model(C.take(inputs, KEYS, f"{name}_"))
        for k, w in weights.items():
            loss += w * C.mse(out[k], labels[f"{name}_{k}"])
    return loss

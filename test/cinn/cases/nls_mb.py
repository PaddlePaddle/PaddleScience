"""NLS-MB optical soliton (examples/NLS-MB/NLS-MB_optical_soliton.py, conf/NLS-MB_soliton.yaml)."""
import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "NLS-MB"
NOTE = (
    "random (t, x) collocation points in [-1,1]^2 instead of the evenly sampled time-x grid; "
    "600 IC/BC points random on t=-1, x=-1, x=1 with analytic labels"
)
NPOINT_INTERIOR = 20000
NPOINT_BC = 600
BATCH_SIZE = NPOINT_INTERIOR + NPOINT_BC
KEYS = ("t", "x")


def build_model():
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=("Eu", "Ev", "pu", "pv", "eta"),
        num_layers=5,
        hidden_size=64,
    )


def analytic_solution(t, x):
    """same formulas as the example"""
    ch = np.cosh(2 * t + 6 * x)
    em, ep = np.exp(-2 * t - 6 * x), np.exp(2 * t + 6 * x)
    return {
        "Eu": 2 * np.cos(2 * t) / ch,
        "Ev": -2 * np.sin(2 * t) / ch,
        "pu": (em - ep) * np.cos(2 * t) / ch**2,
        "pv": -(em - ep) * np.sin(2 * t) / ch**2,
        "eta": (ch**2 - 2) / ch**2,
    }


def build_batch():
    inputs = C.points(KEYS, NPOINT_INTERIOR, -1.0, 1.0)
    n = NPOINT_BC // 3  # lower x bound, initial t, upper x bound
    s = np.random.uniform(-1, 1, [n, 1]).astype("float32")
    c = np.full([n, 1], -1.0, "float32")
    t = np.concatenate([s, c, s])
    x = np.concatenate([c, s, -c])
    inputs.update({"sup_t": paddle.to_tensor(t), "sup_x": paddle.to_tensor(x)})
    labels = {
        k: paddle.to_tensor(v.astype("float32"))
        for k, v in analytic_solution(t, x).items()
    }
    return inputs, labels


EQUATIONS = {}


def prepare(model):
    eq = ppsci.equation.NLSMB(alpha_1=0.5, alpha_2=-1, omega_0=-1, time=True)
    EQUATIONS.update(C.lambdify(eq.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(model, C.take(inputs, KEYS), EQUATIONS)
    return loss + C.data_loss(model(C.take(inputs, KEYS, "sup_")), labels)

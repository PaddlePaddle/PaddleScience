"""heat_exchanger HEDeepONets (examples/heat_exchanger/heat_exchanger.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "heat_exchanger"
NOTE = (
    "random (x, t, qm_h, qm_c) instead of the evenly sampled time-x grid; "
    "qm drawn from [0.05, 2] (example: rand*2) to avoid 1/qm blow-ups"
)
N = 1000  # TRAIN.batch_size, shared by all 4 constraints
BATCH_SIZE = 4 * N
KEYS = ("x", "t", "qm_h", "qm_c")
T_HIN, T_CIN, T_WIN = 10.0, 1.0, 5.5


def build_model():
    return ppsci.arch.HEDeepONets(
        heat_input_keys=("qm_h",),
        cold_input_keys=("qm_c",),
        trunk_input_keys=("x", "t"),
        output_keys=("T_h", "T_c", "T_w"),
        heat_num_loc=1,
        cold_num_loc=1,
        num_features=100,
        branch_num_layers=9,
        trunk_num_layers=6,
        branch_hidden_size=256,
        trunk_hidden_size=128,
        branch_activation="swish",
        trunk_activation="swish",
        use_bias=True,
    )


def _group(prefix, x=None, t=None):
    g = {
        "x": C.uniform([N, 1], 0.0, 1.0, True) if x is None else paddle.full([N, 1], x),
        "t": C.uniform([N, 1], 0.0, 2.0, True) if t is None else paddle.full([N, 1], t),
        "qm_h": C.uniform([N, 1], 0.05, 2.0),
        "qm_c": C.uniform([N, 1], 0.05, 2.0),
    }
    return {prefix + k: v for k, v in g.items()}


def build_batch():
    inputs = _group("")
    inputs.update(_group("left_", x=0.0))
    inputs.update(_group("right_", x=1.0))
    inputs.update(_group("init_", t=0.0))
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    he = ppsci.equation.HeatExchanger(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    EQUATIONS.update(C.lambdify(he.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS))


def compute_loss(model, inputs, labels):
    loss = C.pde_loss(
        model,
        C.take(inputs, KEYS),
        EQUATIONS,
        {"heat_boundary": 1, "cold_boundary": 1, "wall": 20},
    )
    loss += 20 * C.sq_mean(model(C.take(inputs, KEYS, "left_"))["T_h"] - T_HIN)
    loss += 20 * C.sq_mean(model(C.take(inputs, KEYS, "right_"))["T_c"] - T_CIN)
    out = model(C.take(inputs, KEYS, "init_"))
    loss += C.sq_mean(out["T_h"] - T_HIN) + C.sq_mean(out["T_c"] - T_CIN)
    loss += 20 * C.sq_mean(out["T_w"] - T_WIN)
    return loss

"""chip_heat DeepONet chip temperature field (examples/chip_heat/chip_heat.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "chip_heat"
NOTE = (
    "random heat-source / BC vectors (randn) instead of GRF samples, random bc types in {0,1,2,3}, "
    "points on the unit square edges/interior drawn uniformly instead of the 20x20 grid; "
    "harness Adam has no ClipGradByGlobalNorm(1.0)"
)
N = 1000  # TRAIN.batch_size, per constraint
NUM_LOC, BC_NUM_LOC = 324, 76
WEIGHT = 500.0
BATCH_SIZE = 5 * N
# side -> (fixed key, fixed value, derivative key)
SIDES = {
    "down": ("x", 0.0, "x"),
    "left": ("y", 0.0, "y"),
    "right": ("y", 1.0, "y"),
    "top": ("x", 1.0, "x"),
}
IN_KEYS = ("x", "y", "u", "bc", "bc_data", "u_one")


def build_model():
    return ppsci.arch.ChipDeepONets(
        branch_input_keys=("u",),
        BCtype_input_keys=("bc",),
        BC_input_keys=("bc_data",),
        trunk_input_keys=("x", "y"),
        output_keys=("T",),
        num_loc=NUM_LOC,
        bctype_loc=1,
        BC_num_loc=BC_NUM_LOC,
        num_features=400,
        branch_num_layers=9,
        BC_num_layers=9,
        trunk_num_layers=6,
        branch_hidden_size=256,
        BC_hidden_size=256,
        trunk_hidden_size=128,
        branch_activation="swish",
        BC_activation="swish",
        trunk_activation="swish",
        use_bias=True,
    )


def _group(prefix, fixed=None):
    d = {
        "x": C.uniform([N, 1], requires_grad=True),
        "y": C.uniform([N, 1], requires_grad=True),
        "u": C.randn([N, NUM_LOC]),
        "bc": C.randint([N, 1], 4).astype("float32"),
        "bc_data": C.randn([N, BC_NUM_LOC]),
        "u_one": C.randn([N, 1]),
    }
    if fixed is not None:
        key, value = fixed
        d[key] = paddle.full([N, 1], value, dtype="float32")
        d[key].stop_gradient = False
    return {f"{prefix}{k}": v for k, v in d.items()}


def build_batch():
    inputs = _group("interior_")
    for side, (key, value, _) in SIDES.items():
        inputs.update(_group(f"{side}_", (key, value)))
    return inputs, {}


def forward(model, inputs):
    return model(C.take(inputs, IN_KEYS, "interior_"))


def _bc_residual(out, dkey):
    T, u_one, bc = out["T"], out["u_one"], out["bc"]
    dT = C.autodiff.jacobian(T, out[dkey])
    return paddle.where(
        bc == 1,
        dT - u_one,
        paddle.where(
            bc == 0,
            T - u_one,
            paddle.where(
                bc == 2,
                dT + u_one * (T - 1),
                dT + u_one * (T**2 - 1) * (T**2 + 1) * 5.6 / 50000,
            ),
        ),
    )


def compute_loss(model, inputs, labels):
    hessian = C.autodiff.hessian
    loss = 0.0
    for side, (_, _, dkey) in SIDES.items():
        d = C.take(inputs, IN_KEYS, f"{side}_")
        out = {**d, **model(d)}
        # MSELoss('mean') with weight_dict {'chip': 500}
        loss = loss + paddle.mean(WEIGHT * paddle.square(_bc_residual(out, dkey)))
        C.autodiff.clear()
    d = C.take(inputs, IN_KEYS, "interior_")
    T = model(d)["T"]
    res = hessian(T, d["x"]) + hessian(T, d["y"]) + 100 * d["u_one"]
    loss = loss + C.sq_mean(res)
    C.autodiff.clear()
    return loss

"""control_arm 3D linear elasticity, forward analysis (examples/control_arm/forward_analysis.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "control_arm"
NOTE = (
    "forward_analysis.yaml; points sampled uniformly in boxes around the left/right circles and "
    "the approximate arm bounds instead of from control_arm.stl; random unit normals; 'area' set "
    "to a guessed measure / batch and 'sdf' uniform in [0, 1]; constant lr 1e-4 instead of "
    "ExponentialDecay; only the equations that carry a label are evaluated per constraint"
)
LEARNING_RATE = 1.0e-4
NU, E = 0.3, 1.0
LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
TRACTION = (-0.0025, 0.0, 0.0)
N_LEFT, N_RIGHT, N_SURFACE, N_INTERIOR = 128, 256, 4096, 2048
BATCH_SIZE = N_LEFT + N_RIGHT + N_SURFACE + N_INTERIOR
BOUNDS = ((-6.5, 18.5), (-3.0, 3.0), (-2.5, 2.5))  # approximate control_arm.stl bounds
KEYS = ("x", "y", "z")
NORMALS = ("normal_x", "normal_y", "normal_z")
TRACTIONS = ("traction_x", "traction_y", "traction_z")
INTERIOR_EQS = (
    "equilibrium_x",
    "equilibrium_y",
    "equilibrium_z",
    "stress_disp_xx",
    "stress_disp_yy",
    "stress_disp_zz",
    "stress_disp_xy",
    "stress_disp_xz",
    "stress_disp_yz",
)


def build_model():
    kw = dict(
        input_keys=KEYS,
        num_layers=6,
        hidden_size=512,
        activation="silu",
        weight_norm=True,
    )
    disp_net = ppsci.arch.MLP(output_keys=("u", "v", "w"), **kw)
    stress_net = ppsci.arch.MLP(
        output_keys=(
            "sigma_xx",
            "sigma_yy",
            "sigma_zz",
            "sigma_xy",
            "sigma_xz",
            "sigma_yz",
        ),
        **kw,
    )
    return ppsci.arch.ModelList((disp_net, stress_net))


def _box(prefix, n, bounds, measure, normals=False, sdf=False):
    out = {
        f"{prefix}{k}": C.uniform([n, 1], lo, hi, requires_grad=True)
        for k, (lo, hi) in zip(KEYS, bounds)
    }
    out[f"{prefix}area"] = paddle.full([n, 1], measure / n, dtype="float32")
    if normals:
        nv = C.randn([n, 3])
        nv = nv / paddle.linalg.norm(nv, axis=-1, keepdim=True)
        out.update({f"{prefix}{k}": nv[:, i : i + 1] for i, k in enumerate(NORMALS)})
    if sdf:
        out[f"{prefix}sdf"] = C.uniform([n, 1], 0.0, 1.0)
    return out


def build_batch():
    inputs = {}
    left = ((-4.4 - 1.65, -4.4 + 1.65), (-1.65, 1.65), (-1.0, 1.0))
    right = ((15.8 - 2.21, 15.8 + 2.21), (-1.0, 1.0), (-2.21, 2.21))
    inputs.update(_box("left_", N_LEFT, left, 20.0, normals=True))
    inputs.update(_box("right_", N_RIGHT, right, 30.0))
    inputs.update(_box("surface_", N_SURFACE, BOUNDS, 400.0, normals=True))
    inputs.update(_box("interior_", N_INTERIOR, BOUNDS, 150.0, sdf=True))
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    pde = ppsci.equation.LinearElasticity(
        E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
    )
    EQUATIONS.update(C.lambdify(pde.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS, "interior_"))


def _sum_sq(x, area, w=1.0):
    # ppsci.loss.MSELoss("sum") with the geometry "area" weighting
    return paddle.sum(paddle.square(x) * w * area)


def _eq_loss(model, inputs, prefix, targets, weight=1.0):
    x = C.take(inputs, KEYS, prefix)
    data = {**x, **model(x)}
    if f"{prefix}normal_x" in inputs:
        data.update(C.take(inputs, NORMALS, prefix))
    loss = 0.0
    for name, target in targets.items():
        loss = loss + _sum_sq(
            EQUATIONS[name](data) - target, inputs[f"{prefix}area"], weight
        )
    C.autodiff.clear()
    return loss


def compute_loss(model, inputs, labels):
    loss = _eq_loss(model, inputs, "left_", dict(zip(TRACTIONS, TRACTION)))
    out = model(C.take(inputs, KEYS, "right_"))
    loss += sum(_sum_sq(out[k], inputs["right_area"], 1.0) for k in ("u", "v", "w"))
    loss += _eq_loss(model, inputs, "surface_", dict.fromkeys(TRACTIONS, 0.0))
    loss += _eq_loss(
        model,
        inputs,
        "interior_",
        dict.fromkeys(INTERIOR_EQS, 0.0),
        inputs["interior_sdf"],
    )
    return loss

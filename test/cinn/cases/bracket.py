"""bracket 3D linear elasticity of a bracket (examples/bracket/bracket.py, conf/bracket.yaml)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "bracket"
NOTE = (
    "points sampled uniformly in the support/bracket boxes and on the back/front faces instead of "
    "from the STL CSG geometry; surface points get random unit normals; 'area' set to box "
    "measure / batch and 'sdf' weights uniform in [0, half thickness]; constant lr instead of "
    "ExponentialDecay; only the equations that carry a label are evaluated per constraint"
)
NU, E = 0.3, 100.0e9
LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
MU_C = 0.01 * MU
LAMBDA_, MU = LAMBDA_ / MU_C, MU / MU_C
T = -4.0e4 * 1.0 / (1.0e-4 * MU_C)  # SIGMA_NORMALIZATION = L / (D * MU_C)
N_BACK, N_FRONT, N_SURFACE, N_SUPPORT, N_BRACKET = 1024, 128, 4096, 2048, 1024
BATCH_SIZE = N_BACK + N_FRONT + N_SURFACE + N_SUPPORT + N_BRACKET
KEYS = ("x", "y", "z")
NORMALS = ("normal_x", "normal_y", "normal_z")
INTERIOR_EQS = (
    "stress_disp_xx",
    "stress_disp_yy",
    "stress_disp_zz",
    "stress_disp_xy",
    "stress_disp_xz",
    "stress_disp_yz",
    "equilibrium_x",
    "equilibrium_y",
    "equilibrium_z",
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


def _box(prefix, n, bounds, measure, normal=None, sdf_max=None):
    out = {
        f"{prefix}{k}": C.uniform([n, 1], lo, hi, requires_grad=True)
        for k, (lo, hi) in zip(KEYS, bounds)
    }
    out[f"{prefix}area"] = paddle.full([n, 1], measure / n, dtype="float32")
    if normal == "random":
        nv = C.randn([n, 3])
        nv = nv / paddle.linalg.norm(nv, axis=-1, keepdim=True)
        out.update({f"{prefix}{k}": nv[:, i : i + 1] for i, k in enumerate(NORMALS)})
    elif normal is not None:
        out.update(
            {
                f"{prefix}{k}": paddle.full([n, 1], v, "float32")
                for k, v in zip(NORMALS, normal)
            }
        )
    if sdf_max is not None:
        out[f"{prefix}sdf"] = C.uniform([n, 1], 0.0, sdf_max)
    return out


def build_batch():
    inputs = {}
    inputs.update(_box("back_", N_BACK, ((-1, -1), (-1, 1), (-1, 1)), 4.0))
    inputs.update(
        _box(
            "front_",
            N_FRONT,
            ((1, 1), (-1, 1), (-0.1, 0.1)),
            0.4,
            normal=(1.0, 0.0, 0.0),
        )
    )
    inputs.update(
        _box("surface_", N_SURFACE, ((-1, 1), (-1, 1), (-1, 1)), 15.0, normal="random")
    )
    inputs.update(
        _box("support_", N_SUPPORT, ((-1, -0.65), (-1, 1), (-1, 1)), 1.4, sdf_max=0.175)
    )
    inputs.update(
        _box(
            "bracket_", N_BRACKET, ((-0.65, 1), (-1, 1), (-0.1, 0.1)), 0.66, sdf_max=0.1
        )
    )
    return inputs, {}


EQUATIONS = {}


def prepare(model):
    pde = ppsci.equation.LinearElasticity(lambda_=LAMBDA_, mu=MU, dim=3)
    EQUATIONS.update(C.lambdify(pde.equations, model))


def forward(model, inputs):
    return model(C.take(inputs, KEYS, "support_"))


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
    out = model(C.take(inputs, KEYS, "back_"))
    loss = sum(_sum_sq(out[k], inputs["back_area"], 10.0) for k in ("u", "v", "w"))
    loss += _eq_loss(
        model, inputs, "front_", {"traction_x": 0.0, "traction_y": 0.0, "traction_z": T}
    )
    loss += _eq_loss(
        model,
        inputs,
        "surface_",
        {"traction_x": 0.0, "traction_y": 0.0, "traction_z": 0.0},
    )
    for part in ("support_", "bracket_"):
        loss += _eq_loss(
            model, inputs, part, dict.fromkeys(INTERIOR_EQS, 0.0), inputs[f"{part}sdf"]
        )
    return loss

"""bubble two-phase bubble flow, semi-supervised PINN (examples/bubble/bubble.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "bubble"
NOTE = (
    "random (t, x, y) in [0,126]x[0,15]x[0,5] and random labels in [0,1] instead of bubble.mat; "
    "infer runs the psi->(u, v) jacobian transform, so forward enables grad"
)
N_PDE, N_SUP = 228595, 2419
BATCH_SIZE = N_PDE + N_SUP
KEYS = ("t", "x", "y")
LABEL_KEYS = ("u", "v", "p", "phil")


def _mlp(out_key):
    return ppsci.arch.MLP(
        input_keys=KEYS,
        output_keys=(out_key,),
        num_layers=9,
        hidden_size=30,
        activation="tanh",
    )


def _transform_out(in_, out):
    psi = out["psi"]
    return {
        "u": C.autodiff.jacobian(psi, in_["y"]),
        "v": -C.autodiff.jacobian(psi, in_["x"]),
    }


def build_model():
    model_psi = _mlp("psi")
    model_psi.register_output_transform(_transform_out)
    return ppsci.arch.ModelList((model_psi, _mlp("p"), _mlp("phil")))


def _txy(n, prefix=""):
    return {
        f"{prefix}t": C.uniform([n, 1], 0.0, 126.0, requires_grad=True),
        f"{prefix}x": C.uniform([n, 1], 0.0, 15.0, requires_grad=True),
        f"{prefix}y": C.uniform([n, 1], 0.0, 5.0, requires_grad=True),
    }


def build_batch():
    inputs = {**_txy(N_PDE), **_txy(N_SUP, "sup_")}
    labels = {k: C.uniform([N_SUP, 1]) for k in LABEL_KEYS}
    return inputs, labels


def forward(model, inputs):
    with paddle.set_grad_enabled(True):
        out = model(C.take(inputs, KEYS, "sup_"))
    C.autodiff.clear()
    return out


def compute_loss(model, inputs, labels):
    hessian = C.autodiff.hessian
    out = model(C.take(inputs, KEYS, "sup_"))
    loss = C.data_loss(out, labels, LABEL_KEYS)
    C.autodiff.clear()
    d = C.take(inputs, KEYS)
    p = model(d)["p"]
    loss = loss + C.sq_mean(hessian(p, d["x"]) + hessian(p, d["y"]))
    C.autodiff.clear()
    return loss

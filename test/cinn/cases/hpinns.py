"""hpinns holography, aug_lag mode, Adam stage (examples/hpinns/holography.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "hpinns"
NOTE = (
    "random points instead of hpinns_holo_train.mat (N_OBJ objective-region points first, "
    "then domain points in the PML box); the numpy-only terms of functions.py (PML "
    "coefficients, eps mask, obj_func_J) are precomputed per point in build_batch; "
    "lambda=0 and loss weights [1,1,1,1,1] as at the start of aug_lag training; LBFGS stages skipped"
)
N_OBJ, N_PDE = 1000, 9000  # dataset sizes are unknown offline; 'bound' = N_OBJ
BATCH_SIZE = N_OBJ + N_PDE
_INPUTS = tuple(
    [f"x_{f}_{i}" for i in range(1, 7) for f in ("cos", "sin")]
    + ["y", "y_cos_1", "y_sin_1"]
)
F = {}


def _functions():
    if not F:
        path = (
            Path(__file__).resolve().parents[3] / "examples" / "hpinns" / "functions.py"
        )
        spec = importlib.util.spec_from_file_location("hpinns_functions", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        F["m"] = mod
    return F["m"]


def build_model():
    f = _functions()
    nets = []
    for out, tr in (
        ("e_re", f.transform_out_real_part),
        ("e_im", f.transform_out_imaginary_part),
        ("eps", f.transform_out_epsilon),
    ):
        net = ppsci.arch.MLP(
            _INPUTS, (out,), num_layers=4, hidden_size=48, activation="tanh"
        )
        net.register_input_transform(f.transform_in)
        net.register_output_transform(tr)
        nets.append(net)
    return ppsci.arch.ModelList(nets)


def build_batch():
    f = _functions()
    lb = f.l_BOX
    x = np.concatenate(
        [
            np.random.uniform(-0.5, 0.5, [N_OBJ, 1]),
            np.random.uniform(lb[0][0], lb[1][0], [N_PDE, 1]),
        ]
    ).astype("float32")
    y = np.concatenate(
        [
            np.random.uniform(1.0, 2.0, [N_OBJ, 1]),
            np.random.uniform(lb[0][1], lb[1][1], [N_PDE, 1]),
        ]
    ).astype("float32")
    xt, yt = paddle.to_tensor(x), paddle.to_tensor(y)
    pml = f.perfectly_matched_layers(xt, yt)
    labels = {
        k: paddle.to_tensor(v.astype("float32"))
        for k, v in zip(("a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"), pml)
    }
    labels["cond"] = paddle.to_tensor(((y < 0) & (y > -1)).astype("float32"))
    labels["J"] = paddle.to_tensor(f.obj_func_J(yt).astype("float32"))
    labels["lambda_re"] = paddle.zeros([N_PDE, 1])
    labels["lambda_im"] = paddle.zeros([N_PDE, 1])
    xt.stop_gradient = yt.stop_gradient = False
    return {"x": xt, "y": yt}, labels


def compute_loss(model, inputs, labels):
    from ppsci.autodiff import hessian, jacobian

    x, y = inputs["x"], inputs["y"]
    out = model(inputs)
    e_re, e_im = out["e_real"], out["e_imaginary"]
    eps = out["epsilon"] * labels["cond"] + 1 - labels["cond"]
    d = {
        "re_x": jacobian(e_re, x),
        "re_y": jacobian(e_re, y),
        "re_xx": hessian(e_re, x),
        "re_yy": hessian(e_re, y),
        "im_x": jacobian(e_im, x),
        "im_y": jacobian(e_im, y),
        "im_xx": hessian(e_im, x),
        "im_yy": hessian(e_im, y),
    }
    ppsci.autodiff.clear()
    a1, b1, a2, b2, a3, b3, a4, b4 = (
        labels[k] for k in ("a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4")
    )
    omega = 2 * np.pi

    def op(p):
        return (
            a1 * d[p + "_xx"] + a2 * d[p + "_x"] + a3 * d[p + "_yy"] + a4 * d[p + "_y"]
        ) / omega

    def op_b(p):
        return (
            b1 * d[p + "_xx"] + b2 * d[p + "_x"] + b3 * d[p + "_yy"] + b4 * d[p + "_y"]
        ) / omega

    loss_re = (op("re") - op_b("im") + eps * omega * e_re)[N_OBJ:]
    loss_im = (op("im") + op_b("re") + eps * omega * e_im + labels["J"])[N_OBJ:]
    pde = (
        C.sq_mean(loss_re)
        + C.sq_mean(loss_im)
        + paddle.mean(loss_re * labels["lambda_re"])
        + paddle.mean(loss_im * labels["lambda_im"])
    )
    # objective loss on the first N_OBJ points
    half = paddle.full([], 0.5)
    f1 = paddle.heaviside((x + 0.5) * (0.5 - x), half)
    f2 = paddle.heaviside((y - 1) * (2 - y), half)
    j = e_re[:N_OBJ] ** 2 + e_im[:N_OBJ] ** 2 - f1[:N_OBJ] * f2[:N_OBJ]
    return pde + C.sq_mean(j)

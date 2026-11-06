"""EPNN elasto-plastic surrogate (examples/epnn/epnn.py + functions.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "epnn"
NOTE = (
    "three Epnn nets + trainable gkratio in one Layer with the example's input/output "
    "transforms (ModelList semantics); itrain guessed as 2000 (depends on .dat size); "
    "normalisation stats (min/range) are synthetic label tensors; per-net lrs "
    "0.001/0.001/0.01/0.001 kept via param learning_rate multipliers on one Adam, "
    "ExponentialDecay (decay_steps=1e7) treated as constant"
)
ITRAIN = 2000
BATCH_SIZE = ITRAIN
LEARNING_RATE = 1e-3
NX_STATE, NY_STATE, NX_STRESS, NY_STRESS = 13, 4, 13, 3
STATE_IDX = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12]


def _transform_in(inp):
    return {k: paddle.squeeze(v, axis=0) for k, v in inp.items()}


def _transform_out(inp, out):
    out.update(inp)
    return out


class EPNN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        act = "leaky_relu"
        self.elasto = ppsci.arch.Epnn(
            ("state_x",),
            ("out_state_elasto",),
            (NX_STATE, 60, 60, 60, NY_STATE - 3),
            (act,) * 5,
            0.0,
        )
        self.plastic = ppsci.arch.Epnn(
            ("state_x",),
            ("out_state_plastic",),
            (NX_STATE, 75, 75, 75, 75, NY_STATE - 1),
            (act,) * 6,
            0.0,
        )
        self.stress = ppsci.arch.Epnn(
            ("state_x_f",),
            ("out_stress",),
            (NX_STRESS + NY_STATE - 6, 60, 60, 60, 1),
            (act,) * 5,
            0.0,
        )
        self.elasto.register_input_transform(_transform_in)
        self.plastic.register_input_transform(_transform_in)
        self.stress.register_input_transform(self._transform_in_stress)
        self.stress.register_output_transform(_transform_out)
        self.gkratio = self.create_parameter(
            [1, 1], default_initializer=paddle.nn.initializer.Constant(0.45)
        )
        for p in self.stress.parameters():
            p.optimize_attr["learning_rate"] = 10.0  # lr 0.01 vs base 0.001

    def _transform_in_stress(self, inp):
        e = self.elasto(inp)["out_state_elasto"].detach().clone()
        t = _transform_in(inp)
        idx = paddle.to_tensor(STATE_IDX)
        t["state_x_f"] = paddle.concat(
            [e, paddle.index_select(t["state_x"], idx, axis=1)], axis=1
        )
        return t

    def forward(self, x):  # ppsci.arch.ModelList
        out = {}
        for m in (self.elasto, self.plastic, self.stress):
            out.update(m(x))
        return out


def build_model():
    return EPNN()


def build_batch():
    # NamedArrayDataset with batch_size=1 yields [1, itrain, dim] per step
    inputs = {
        "state_x": C.uniform([1, ITRAIN, NX_STATE], -1.0, 1.0),
        "state_y": C.uniform([1, ITRAIN, NY_STATE], -1.0, 1.0),
        "stress_x": C.uniform([1, ITRAIN, NX_STRESS], -1.0, 1.0),
        "stress_y": C.uniform([1, ITRAIN, NY_STRESS], -1.0, 1.0),
    }
    # common_param, named as unpacked in functions.loss_func
    labels = {
        "min_elasto": C.uniform([NY_STATE], -1.0, 0.0),
        "range_elasto": C.uniform([NY_STATE], 0.5, 2.0),
        "min_plastic": C.uniform([NY_STRESS], -1.0, 0.0),
        "range_plastic": C.uniform([NY_STRESS], 0.5, 2.0),
        "min_stress": C.uniform([3], -1.0, 0.0),
        "range_stress": C.uniform([3], 0.5, 2.0),
    }
    return inputs, labels


def compute_loss(model, inputs, labels):
    out, L = model(inputs), labels
    mae = paddle.nn.functional.l1_loss
    target_plastic = out["state_y"][:, 1:4]
    loss_elasto = mae(out["out_state_elasto"], out["state_y"][:, 0:1])
    loss_plastic = mae(out["out_state_plastic"], target_plastic)
    stress = out["out_stress"]
    oneten_state = paddle.ones([3, 1])
    oneten_stress = paddle.ones([out["stress_y"].shape[0], out["stress_y"].shape[1]])
    dstrain_real = (out["state_x"][:, 10:] + 1.0) * L["range_stress"] / 2.0 + L[
        "min_stress"
    ]
    dstrainpl_real = (target_plastic + 1.0) * L["range_elasto"][1:4] / 2.0 + L[
        "min_elasto"
    ][1:4]
    dstrainel = dstrain_real - dstrainpl_real
    mu = model.gkratio * stress[:, 0:1]
    mu_dstrainel = 2.0 * mu * dstrainel
    stress_dstrainel = (stress[:, 0:1] - 2.0 / 3.0 * mu) * (
        paddle.matmul(dstrainel, oneten_state) * oneten_stress
    )
    pred = (
        2.0 * (mu_dstrainel + stress_dstrainel - L["min_plastic"]) / L["range_plastic"]
        - 1.0
    )
    return loss_elasto + loss_plastic + mae(pred, out["stress_y"])

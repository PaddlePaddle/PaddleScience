"""IFM-MLP on tox21 (examples/ifm/ifm.py, data_label=tox21, embed IFM)."""
import paddle
from cases import _common as C
from paddle.nn import BCEWithLogitsLoss

import ppsci

EXAMPLE = "ifm"
NOTE = (
    "tox21 input dim taken as 249 (comment in IFMMoeDataset; real value depends on "
    "feature filtering); pos_weights passed as a label tensor; Adam weight_decay "
    "reproduced via per-param L2Decay"
)
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
IN_DIM = 249
N_TASKS = 12
HP = dict(
    d_out=40,
    dropout=0.040875279882956944,
    l2=4.482713460277411e-06,
    omega0=0.6124886445092693,
    omega1=0.7820986807482367,
    sigma=13.846474804138776,
)


def build_model():
    model = ppsci.arch.IFMMLP(
        input_keys=("x",),
        output_keys=("pred",),
        hidden_units=[512, 512, 512],
        embed_name="IFM",
        inputs=IN_DIM,
        outputs=N_TASKS,
        d_out=HP["d_out"],
        sigma=HP["sigma"],
        dp_ratio=HP["dropout"],
        reg=False,
        first_omega_0=HP["omega0"],
        hidden_omega_0=HP["omega1"],
    )
    for p in model.parameters():
        p.regularizer = paddle.regularizer.L2Decay(HP["l2"])
    return model


def build_batch():
    mask = (C.uniform([BATCH_SIZE, N_TASKS]) < 0.85).astype("float32")
    y = (C.uniform([BATCH_SIZE, N_TASKS]) < 0.1).astype("float32") * mask  # nan -> 0
    pos_weight = C.uniform([N_TASKS], 5.0, 40.0)  # (N - pos) / pos over the dataset
    return {"x": C.randn([BATCH_SIZE, IN_DIM])}, {
        "y": y,
        "mask": mask,
        "pos_weight": pos_weight,
    }


def compute_loss(model, inputs, labels):
    out = model(inputs)
    loss_fn = BCEWithLogitsLoss(reduction="none", pos_weight=labels["pos_weight"])
    loss = loss_fn(out["pred"], labels["y"]) * (labels["mask"] != 0).astype("float32")
    return loss.mean()

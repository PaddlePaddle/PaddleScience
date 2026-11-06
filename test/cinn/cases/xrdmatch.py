"""xrdmatch VGG16-1D FlexMatch semi-supervised XRD classifier (examples/xrdmatch/main.py)."""
import paddle
import paddle.nn.functional as F
from cases import _common as C

import ppsci

EXAMPLE = "xrdmatch"
NOTE = (
    "synthetic XRD spectra [*,1,4501] in [0,1], no weak/strong numpy augmentation; "
    "FlexMatch mask uses fixed p_cutoff=0.95 (example adapts thresholds from host-side "
    "numpy class stats after step 1); weak-view no_grad forward -> detach; "
    "`if mask.sum()>0` -> paddle.where; harness Adam instead of AdamW(wd=0.01)+cosine"
)
BATCH_SIZE = 32  # labeled; unlabeled batch = 3 * 32 for each of weak/strong views
ULB_BATCH = 3 * BATCH_SIZE
LEN = 4501
LEARNING_RATE = 3e-4
T, P_CUTOFF, LAMBDA_U = 0.5, 0.95, 1.0


def build_model():
    model = ppsci.arch.VGG(in_channel=1, num_classes=2)
    model.eval()
    with paddle.no_grad():  # VGG builds its classifier lazily on first call
        model(paddle.zeros([1, 1, LEN]))
    model.train()
    return model


def _spectra(n):
    x = C.uniform([n, 1, LEN])
    return paddle.where(x < 0.1, paddle.zeros_like(x), x)  # data_zero


def build_batch():
    inputs = {
        "x_lb": _spectra(BATCH_SIZE),
        "x_ulb_w": _spectra(ULB_BATCH),
        "x_ulb_s": _spectra(ULB_BATCH),
    }
    return inputs, {"y_lb": C.randint([BATCH_SIZE], 2)}


def forward(model, inputs):
    return model(inputs["x_lb"])


def compute_loss(model, inputs, labels):
    logits_lb = model(inputs["x_lb"])["logits"]
    logits_w = model(inputs["x_ulb_w"])["logits"].detach()
    logits_s = model(inputs["x_ulb_s"])["logits"]
    loss_lb = F.cross_entropy(logits_lb, labels["y_lb"])
    # gen_pseudo_label (hard labels)
    scaled = logits_w / T
    probs = F.softmax(scaled - paddle.max(scaled, axis=-1, keepdim=True), axis=-1)
    pseudo_label = paddle.argmax(probs, axis=-1)
    mask = (paddle.max(probs, axis=-1) >= P_CUTOFF).astype("float32")
    loss_ulb = F.cross_entropy(logits_s, pseudo_label, reduction="none").squeeze(-1)
    loss_ulb = paddle.where(
        mask.sum() > 0, (loss_ulb * mask).mean(), paddle.zeros([], "float32")
    )
    return loss_lb + LAMBDA_U * loss_ulb

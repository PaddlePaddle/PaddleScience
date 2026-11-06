"""topopt TopOptNN (UNetEx variant) for SIMP topology optimization (examples/topopt/topopt.py)."""
import sys
from pathlib import Path

import paddle
from cases import _common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "topopt"))
from topoptmodel import TopOptNN  # noqa: E402

EXAMPLE = "topopt"
NOTE = (
    "synthetic SIMP iterations [B,100,40,40] in [0,1] and binary targets [B,1,40,40]; "
    "channel sampler fixed to k=5 (example: Poisson(5) etc., Python RNG would be frozen "
    "at trace time); no D4 augmentation; Adam eps default 1e-8 instead of 1e-7"
)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3 / (1 + 30 // 15)
VOL_COEFF = 1


def build_model():
    return TopOptNN(
        in_channel=2,
        out_channel=1,
        kernel_size=3,
        filters=[16, 32, 64],
        layers=2,
        channel_sampler=lambda: 5,
    )


def build_batch():
    target = paddle.cast(C.uniform([BATCH_SIZE, 1, 40, 40]) > 0.5, "float32")
    return {"input": C.uniform([BATCH_SIZE, 100, 40, 40])}, {"output": target}


def compute_loss(model, inputs, labels):
    label_true = labels["output"].reshape((-1, 1))
    label_pred = model(inputs)["output"].reshape((-1, 1))
    conf_loss = paddle.mean(
        paddle.nn.functional.log_loss(label_pred, label_true, epsilon=1.0e-7)
    )
    vol_loss = paddle.square(paddle.mean(label_true - label_pred))
    return conf_loss + VOL_COEFF * vol_loss

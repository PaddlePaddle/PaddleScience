"""UTAE U-TAE semantic segmentation on PASTIS (examples/UTAE/train_semantic.py defaults)."""
import sys
from pathlib import Path

import paddle
from cases import _common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "UTAE"))
from src.backbones.utae import UTAE  # noqa: E402

EXAMPLE = "UTAE"
NOTE = (
    "ported from examples/UTAE/benchmark_cinn.py; synthetic PASTIS batch: series "
    "[B,T=61,C=10,128,128] randn (never all-zero, so no padded frames) and dates "
    "[B,61] int64 in [0,365); CrossEntropyLoss(ignore_index=-1) on [B,20,128,128] "
    "logits; uses the local utae.py/ltae.py edits (GroupNorm rank-3, paddle.where "
    "instead of boolean masked assignment). Known CINN issues (paddle 3.2.0): infer hits an "
    "illegal memory access in a CINN kernel inside the pd_op.if of "
    "Temporal_Aggregator's `pad_mask.any()` branch; train runs but its loss drifts "
    "from step 2 (first 5e-6, final 5e-2 rel diff), as in the old benchmark"
)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
IMG_SIZE = (128, 128)
NUM_CHANNELS = 10
SEQ_LEN = 61
NUM_CLASSES = 20
IGNORE_INDEX = -1


def build_model():
    return UTAE(
        input_dim=NUM_CHANNELS,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, NUM_CLASSES],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        pad_value=0,
        padding_mode="reflect",
    )


def build_batch():
    inputs = {
        "series": C.randn([BATCH_SIZE, SEQ_LEN, NUM_CHANNELS, *IMG_SIZE]),
        "dates": C.randint([BATCH_SIZE, SEQ_LEN], 365),
    }
    labels = {"target": C.randint([BATCH_SIZE, *IMG_SIZE], NUM_CLASSES)}
    return inputs, labels


def forward(model, inputs):
    return model(inputs["series"], batch_positions=inputs["dates"])


def compute_loss(model, inputs, labels):
    out = forward(model, inputs)  # (B, NUM_CLASSES, H, W)
    return paddle.nn.functional.cross_entropy(
        out.transpose([0, 2, 3, 1]).reshape([-1, NUM_CLASSES]),
        labels["target"].reshape([-1]),
        ignore_index=IGNORE_INDEX,
    )

"""tgcn traffic forecasting, PEMSD8 (examples/tgcn/run.py, conf/run.yaml)."""
import numpy as np
import paddle.nn.functional as F
from cases import _common as C

from ppsci.arch.tgcn import TGCN

EXAMPLE = "tgcn"
NOTE = (
    "random row-normalized 170-node graph (dist.csv unavailable, same as benchmark_cinn.py); "
    "graph_conv uses functional F.dropout(0.25), set to 0 here like the harness does "
    "for nn.Dropout"
)
BATCH_SIZE = 64
NUM_NODES, INPUT_LEN, LABEL_LEN = 170, 12, 12
LEARNING_RATE = 0.01


def build_adj():
    rng = np.random.default_rng(2024)
    adj = (rng.random((NUM_NODES, NUM_NODES)) < 0.05).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj / adj.sum(axis=-1, keepdims=True)  # reduce="mean"


def build_model():
    model = TGCN(
        ("input",),
        ("label",),
        adj=build_adj(),
        in_dim=1,
        emb_dim=32,
        hidden=64,
        gc_layer=2,
        tc_layer=2,
        k_s=3,
        dropout=0.25,
        alpha=0.1,
        input_len=INPUT_LEN,
        label_len=LABEL_LEN,
    )
    for conv in (model.sc1_conv, model.sc2_conv):
        conv.dropout = 0.0
    return model


def build_batch():
    return (
        {"input": C.randn([BATCH_SIZE, INPUT_LEN, NUM_NODES, 1])},
        {"label": C.randn([BATCH_SIZE, LABEL_LEN, NUM_NODES, 1])},
    )


def compute_loss(
    model, inputs, labels
):  # ppsci.loss.L1Loss("mean"): sum over axis 1, then mean
    out = model(inputs)["label"]
    return F.l1_loss(out, labels["label"], "none").sum(axis=1).mean()

"""lorenz PhysformerGPT2 on Koopman embeddings (examples/lorenz/train_transformer.py)."""
from cases import _common as C

import ppsci

EXAMPLE = "lorenz"
NOTE = (
    "embeds are synthetic randn [B, 63, 32]: the frozen LorenzEmbedding encoder runs inside the "
    "dataset, not in the trained graph; Adam without grad clip 0.1 / wd 1e-8 / cosine "
    "restarts; infer is the example's eval path (generate: 255 autoregressive steps "
    "from embeds[:, :1])"
)
BATCH_SIZE = 16
BLOCK_SIZE = 64  # TRAIN_BLOCK_SIZE


def build_model():
    return ppsci.arch.PhysformerGPT2(
        ("embeds",),
        ("pred_embeds",),
        num_layers=4,
        num_ctx=64,
        embed_size=32,
        num_heads=4,
    )


def build_batch():
    x = C.randn([BATCH_SIZE, BLOCK_SIZE, 32])
    return {"embeds": x[:, :-1]}, {"pred_embeds": x[:, 1:]}


def compute_loss(model, inputs, labels):  # ppsci.loss.MSELoss("mean")
    return C.data_loss(model(inputs), labels)

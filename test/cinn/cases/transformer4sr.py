"""transformer4sr symbolic regression transformer (examples/transformer4sr/transformer4sr.py)."""
import paddle
from cases import _common as C

import ppsci

EXAMPLE = "transformer4sr"
NOTE = (
    "synthetic batch: values randn [B, 50, 7, 1], token targets = 1 + random tokens in "
    "[2, 20) of random length 1..30, zero padded to 31; plain Adam(lr=1e-3) instead of "
    "Adam(0.9, 0.98, 1e-9) with Noam warmup; dropout=0.25 is nn.Dropout (zeroed)"
)
BATCH_SIZE = 512
SEQ_LEN = 30  # seq_length_max
NUM_VAR_MAX = 7  # len(response_variable)
VOCAB_SIZE = 18 + 2  # len(vocab_library) + 2
SAMPLING_TIMES = 50


def build_model():
    return ppsci.arch.Transformer(
        ("input", "target_seq"),
        ("output",),
        d_model=256,
        heads=4,
        num_layers_enc=4,
        num_layers_dec=8,
        act="relu",
        dropout=0.25,
        num_var_max=NUM_VAR_MAX,
        vocab_size=VOCAB_SIZE,
        seq_length=SEQ_LEN,
    )


def build_batch():
    tokens = C.randint([BATCH_SIZE, SEQ_LEN + 1], VOCAB_SIZE, low=2)
    length = C.randint([BATCH_SIZE, 1], SEQ_LEN + 1, low=1)
    pos = paddle.arange(SEQ_LEN + 1, dtype="int64").unsqueeze(0)
    targets = paddle.where(pos <= length, tokens, paddle.zeros_like(tokens))
    targets[:, 0] = 1  # start token
    inputs = {
        "input": C.randn([BATCH_SIZE, SAMPLING_TIMES, NUM_VAR_MAX, 1]),
        "target_seq": targets[:, :-1],
    }
    return inputs, {"output": targets[:, 1:]}


def compute_loss(
    model, inputs, labels
):  # functions_loss_metric.cross_entropy_loss_func
    loss_fn = paddle.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    return loss_fn(model(inputs)["output"], labels["output"])

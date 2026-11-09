"""fourcastnet AFNONet pretrain on ERA5 (examples/fourcastnet/conf/fourcastnet_pretrain.yaml)."""
from cases import _common as C

import ppsci

EXAMPLE = "fourcastnet"
NOTE = (
    "synthetic 20-channel 720x1440 input/output (pretrain IMG_H/IMG_W, TRAIN.batch_size=1); "
    "AFNONet defaults (patch 8x8, embed_dim 768, depth 12, 8 blocks); MSE on output; "
    "train is kept but fails under CINN (paddle 3.2.0) while compiling the backward: "
    "'op [cinn_op.concat] kernel output args (0) defs should equal op outputs (1)' "
    "(pd_op_to_kernel_pass.cc), likely from the set_value_with_tensor_grad of the "
    "spectral-filter slice assignments; this is why the old benchmark ran forward-only"
)
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
IMG_SIZE = (720, 1440)
NUM_CHANNELS = 20


def build_model():
    return ppsci.arch.AFNONet(
        input_keys=("input",),
        output_keys=("output",),
        img_size=IMG_SIZE,
        in_channels=NUM_CHANNELS,
        out_channels=NUM_CHANNELS,
    )


def build_batch():
    shape = [BATCH_SIZE, NUM_CHANNELS, *IMG_SIZE]
    return {"input": C.randn(shape)}, {"output": C.randn(shape)}


def compute_loss(model, inputs, labels):
    return C.data_loss(model(inputs), labels)

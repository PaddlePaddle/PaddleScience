"""unetformer UNetFormer (resnet18 + global-local attention decoder) on Vaihingen
(examples/unetformer/config/vaihingen/unetformer.py)."""
import sys
from pathlib import Path

import paddle
from cases import _common as C

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "unetformer"))
from geoseg.losses.useful_loss import UnetFormerLoss  # noqa: E402
from geoseg.models.UNetFormer import UNetFormer  # noqa: E402

EXAMPLE = "unetformer"
NOTE = (
    "ported from examples/unetformer/benchmark_cinn.py; resnet18 backbone built with "
    "pretrained=False (random init, ImageNet weights need network); drop_path is "
    "already 0 in every Block (nn.Identity), Dropout/Dropout2D zeroed by the runner; "
    "synthetic image [B,3,512,512] randn and mask in [0,6); UnetFormerLoss "
    "(soft CE + Dice, +0.4 aux CE) with ignore_index=6; Adam lr 1e-4 as the old "
    "benchmark (config uses AdamW 6e-4 + cosine); uses the local view->reshape edit; "
    "the old benchmark kept the loss in dygraph, here it is traced too, so "
    "paddle.pir.Value.view is aliased to reshape (the example's paddle_utils only "
    "patches paddle.Tensor.view) for DiceLoss.view(bs, -1)"
)
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = (512, 512)
NUM_CHANNELS = 3
NUM_CLASSES = 6
IGNORE_INDEX = NUM_CLASSES


def _value_view(self, *args, **kwargs):
    """paddle_utils.add_tensor_methods' Tensor.view, for static-graph Values:
    pir Value.view only works in dygraph and DiceLoss calls x.view(bs, -1)."""
    if args:
        shape = (
            args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        )
        return paddle.reshape(self, list(shape))
    return paddle.reshape(self, shape=list(kwargs.values())[0])


paddle.pir.Value.view = _value_view
LOSS = UnetFormerLoss(ignore_index=IGNORE_INDEX)  # stateless, train mode


def build_model():
    return UNetFormer(num_classes=NUM_CLASSES, pretrained=False)


def build_batch():
    return (
        {"image": C.randn([BATCH_SIZE, NUM_CHANNELS, *IMG_SIZE])},
        {"mask": C.randint([BATCH_SIZE, *IMG_SIZE], NUM_CLASSES)},
    )


def forward(model, inputs):
    return model(inputs["image"])


def compute_loss(model, inputs, labels):
    # training mode returns (main logits, aux logits)
    return LOSS(forward(model, inputs), labels["mask"])

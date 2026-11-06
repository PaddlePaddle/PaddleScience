"""CNN_UTS ResNet18 regression of tensile strength from micrographs (examples/CNN_UTS/main.py)."""
import paddle
from cases import _common as C

EXAMPLE = "CNN_UTS"
NOTE = (
    "resnet18(pretrained=False) instead of ImageNet weights (no download), fc -> Linear(512,1); "
    "synthetic normalized images [B,3,224,224] without random flips; float32 UTS labels in "
    "[200,600] MPa"
)
BATCH_SIZE = 32
LEARNING_RATE = 0.0009761248347350309


def build_model():
    model = paddle.vision.models.resnet18(pretrained=False)
    model.fc = paddle.nn.Linear(model.fc.weight.shape[0], 1)
    return model


def build_batch():
    images = C.randn([BATCH_SIZE, 3, 224, 224])
    return {"images": images}, {"uts": C.uniform([BATCH_SIZE], 200.0, 600.0)}


def forward(model, inputs):
    return model(inputs["images"])


def compute_loss(model, inputs, labels):
    outputs = model(inputs["images"])
    return paddle.nn.functional.mse_loss(outputs.squeeze(), labels["uts"])

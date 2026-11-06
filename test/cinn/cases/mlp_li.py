"""Li-ion electrode property MLP (examples/MLP_LI/MLP_LI.py, initial fit_model)."""
import paddle.nn as nn
from cases import _common as C

EXAMPLE = "MLP_LI"
NOTE = (
    "initial fit_model config (40/30/15, dropout 0.2 -> disabled by harness); "
    "full-batch train set guessed as 4000 rows x 64 features (PCA(0.99)+10 dims "
    "depends on the CSV); Adam lr 1e-4 instead of centered RMSProp(momentum 0.9)"
)
BATCH_SIZE = 4000
IN_DIM = 64
LEARNING_RATE = 1e-4


def build_model():
    return nn.Sequential(
        nn.Linear(IN_DIM, 40),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(40, 30),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(30, 15),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(15, 3),
        nn.Sigmoid(),
    )


def build_batch():
    # MinMax-scaled features and targets
    return {"x": C.uniform([BATCH_SIZE, IN_DIM])}, {"y": C.uniform([BATCH_SIZE, 3])}


def forward(model, inputs):
    return model(inputs["x"])


def compute_loss(model, inputs, labels):
    return C.mse(model(inputs["x"]), labels["y"])

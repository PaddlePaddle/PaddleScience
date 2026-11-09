"""phylstm DeepPhyLSTM model_type=2, Bouc-Wen SDOF (examples/phylstm/phylstm2.py, phylstm2.yaml)."""
import numpy as np
import paddle
import paddle.nn.functional as F
from cases import _common as C

import ppsci

EXAMPLE = "phylstm"
NOTE = (
    "synthetic Bouc-Wen records (100 x 1501 pts @ 50 Hz in data_boucwen.mat): 8 train "
    "records (0.8 of 10) + 50 collocation records, phi = exact FD matrix repeated 50x "
    "[50,1501,1501]; random train/val shuffle omitted"
)
BATCH_SIZE = 1  # one epoch-batch = 8 measured + 50 collocation sequences
N_T, DT = 1501, 0.02
N_TRAIN, N_COLLOC = 8, 50


def build_model():
    return ppsci.arch.DeepPhyLSTM(1, 1, 100, 2)


def _phi_t0():  # finite-difference d/dt matrix, as in phylstm2.py
    n = N_T
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi3 = np.concatenate([np.zeros([n - 3]), np.array([1 / 2, -2, 3 / 2])])
    phi = np.concatenate([phi1[None], temp1 + temp2, phi3[None]], axis=0) / DT
    return phi.reshape([1, n, n]).astype("float32")


def build_batch():
    ag_c = 0.3 * C.randn([N_COLLOC, N_T, 1])
    inputs = {
        "ag": ag_c[:N_TRAIN].clone(),
        "eta": 0.05 * C.randn([N_TRAIN, N_T, 1]),
        "eta_t": 0.2 * C.randn([N_TRAIN, N_T, 1]),
        "lift": -ag_c,
        "ag_c": ag_c,
        "phi": paddle.to_tensor(np.repeat(_phi_t0(), N_COLLOC, axis=0)),
    }
    return inputs, {}


def forward(model, inputs):
    return model({k: inputs[k] for k in ("ag", "ag_c", "phi")})


def compute_loss(model, inputs, labels):  # functions.train_loss_func2
    out = model({k: inputs[k] for k in ("ag", "ag_c", "phi")})
    eta, lift = inputs["eta"], inputs["lift"]
    loss_u = F.mse_loss(eta, out["eta_pred"])
    loss_udot = F.mse_loss(inputs["eta_t"], out["eta_dot_pred"])
    loss_ut_c = F.mse_loss(out["eta_t_pred_c"], out["eta_dot_pred_c"])
    ones = paddle.ones([lift.shape[0], 1, eta.shape[2]], dtype="float32")
    loss_e = F.mse_loss(paddle.matmul(lift, ones), out["lift_pred_c"])
    return paddle.square(loss_u + loss_udot + loss_ut_c + loss_e)

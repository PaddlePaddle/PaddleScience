# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def split_tensors(
    *tensors: List[np.array], ratio: float
) -> Tuple[List[np.array], List[np.array]]:
    """Split tensors to two parts.

    Args:
      tensors (List[np.array]): Non-empty tensor list.
      ratio (float): Split ratio. For example, tensor list A is split to A1 and A2. len(A1) / len(A) = ratio.
    Returns:
      Tuple[List[np.array], List[np.array]]: Splited tensors.
    """
    if len(tensors) == 0:
        raise ValueError("Tensors shouldn't be empty.")

    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        if len(tensor) != count:
            raise ValueError("The size of tensor should be same.")
        x = int(len(tensor) * ratio)
        split1.append(tensor[:x])
        split2.append(tensor[x:])

    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2


def predict_and_save_plot(
    x: np.ndarray, y: np.ndarray, index: int, solver: ppsci.solver.Solver, plot_dir: str
):
    """Make prediction and save visulization of result.

    Args:
      x (np.ndarray): Input of test dataset.
      y (np.ndarray): Output of test dataset.
      index (int): Index of data to visuliaze.
      solver (ppsci.solver.Solver): Trained slover.
      plot_dir (str): Directory to save plot.
    """
    min_u = np.min(y[index, 0, :, :])
    max_u = np.max(y[index, 0, :, :])

    min_v = np.min(y[index, 1, :, :])
    max_v = np.max(y[index, 1, :, :])

    min_p = np.min(y[index, 2, :, :])
    max_p = np.max(y[index, 2, :, :])

    output = solver.predict({"input": x}, return_numpy=True)
    pred_y = output["output"]
    error = np.abs(y - pred_y)

    min_error_u = np.min(error[index, 0, :, :])
    max_error_u = np.max(error[index, 0, :, :])

    min_error_v = np.min(error[index, 1, :, :])
    max_error_v = np.max(error[index, 1, :, :])

    min_error_p = np.min(error[index, 2, :, :])
    max_error_p = np.max(error[index, 2, :, :])

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.subplot(3, 3, 1)
    plt.title("OpenFOAM", fontsize=18)
    plt.imshow(
        np.transpose(y[index, 0, :, :]),
        cmap="jet",
        vmin=min_u,
        vmax=max_u,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.ylabel("Ux", fontsize=18)
    plt.subplot(3, 3, 2)
    plt.title("DeepCFD", fontsize=18)
    plt.imshow(
        np.transpose(pred_y[index, 0, :, :]),
        cmap="jet",
        vmin=min_u,
        vmax=max_u,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.subplot(3, 3, 3)
    plt.title("Error", fontsize=18)
    plt.imshow(
        np.transpose(error[index, 0, :, :]),
        cmap="jet",
        vmin=min_error_u,
        vmax=max_error_u,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")

    plt.subplot(3, 3, 4)
    plt.imshow(
        np.transpose(y[index, 1, :, :]),
        cmap="jet",
        vmin=min_v,
        vmax=max_v,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.ylabel("Uy", fontsize=18)
    plt.subplot(3, 3, 5)
    plt.imshow(
        np.transpose(pred_y[index, 1, :, :]),
        cmap="jet",
        vmin=min_v,
        vmax=max_v,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.subplot(3, 3, 6)
    plt.imshow(
        np.transpose(error[index, 1, :, :]),
        cmap="jet",
        vmin=min_error_v,
        vmax=max_error_v,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")

    plt.subplot(3, 3, 7)
    plt.imshow(
        np.transpose(y[index, 2, :, :]),
        cmap="jet",
        vmin=min_p,
        vmax=max_p,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.ylabel("p", fontsize=18)
    plt.subplot(3, 3, 8)
    plt.imshow(
        np.transpose(pred_y[index, 2, :, :]),
        cmap="jet",
        vmin=min_p,
        vmax=max_p,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.subplot(3, 3, 9)
    plt.imshow(
        np.transpose(error[index, 2, :, :]),
        cmap="jet",
        vmin=min_error_p,
        vmax=max_error_p,
        origin="lower",
        extent=[0, 260, 0, 120],
    )
    plt.colorbar(orientation="horizontal")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        os.path.join(PLOT_DIR, f"cfd_{index}.png"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)

    OUTPUT_DIR = "./output_deepCFD/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize datasets
    DATASET_PATH = "./datasets/"
    with open(os.path.join(DATASET_PATH, "dataX.pkl"), "rb") as file:
        x = pickle.load(file)
    with open(os.path.join(DATASET_PATH, "dataY.pkl"), "rb") as file:
        y = pickle.load(file)

    # slipt dataset to train dataset and test datatset
    SLIPT_RATIO = 0.7
    train_dataset, test_dataset = split_tensors(x, y, ratio=SLIPT_RATIO)
    train_x, train_y = train_dataset
    test_x, test_y = test_dataset

    # initialize parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    KERNEL_SIZE = 5
    FILTERS = (8, 16, 32, 32)
    BATCH_NORM = False
    WEIGHT_NORM = False
    WEIGHT_DECAY = 0.005
    BATCH_SIZE = 64

    # initialize model
    model = ppsci.arch.UNetEx(
        "input",
        "output",
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        FILTERS,
        weight_norm=WEIGHT_NORM,
        batch_norm=BATCH_NORM,
    )

    # the shape of x and y is [SAMPLE_SIZE, CHANNEL_SIZE, X_SIZE, Y_SIZE]
    SAMPLE_SIZE = 981
    CHANNEL_SIZE = 3
    X_SIZE = 172
    Y_SIZE = 79

    CHANNELS_WEIGHTS = np.reshape(
        np.sqrt(
            np.mean(
                np.transpose(y, (0, 2, 3, 1)).reshape(
                    (SAMPLE_SIZE * X_SIZE * Y_SIZE, CHANNEL_SIZE)
                )
                ** 2,
                axis=0,
            )
        ),
        (1, -1, 1, 1),
    )

    # define loss
    def loss_expr(
        output_dict: Dict[str, np.ndarray],
        label_dict: Dict[str, np.ndarray] = None,
        weight_dict: Dict[str, np.ndarray] = None,
    ) -> float:
        output = output_dict["output"]
        y = label_dict["output"]
        loss_u = (output[:, 0:1, :, :] - y[:, 0:1, :, :]) ** 2
        loss_v = (output[:, 1:2, :, :] - y[:, 1:2, :, :]) ** 2
        loss_p = (output[:, 2:3, :, :] - y[:, 2:3, :, :]).abs()
        loss = (loss_u + loss_v + loss_p) / CHANNELS_WEIGHTS
        return loss.sum()

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": train_x},
                "label": {"output": train_y},
            },
            "batch_size": BATCH_SIZE,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
        },
        ppsci.loss.FunctionalLoss(loss_expr),
        name="sup_constraint",
    )

    # maunally build constraint
    constraint = {sup_constraint.name: sup_constraint}

    # set training hyper-parameters
    EPOCHS = 1000
    LEARNING_RATE = 0.001

    # initialize Adam optimizer
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)(model)

    # manually build validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": {"input": test_x},
            "label": {"output": test_y},
        },
        "batch_size": 8,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    def metric_expr(
        output_dict: Dict[str, np.ndarray],
        label_dict: Dict[str, np.ndarray] = None,
        weight_dict: Dict[str, np.ndarray] = None,
    ) -> Dict[str, float]:
        output = output_dict["output"]
        y = label_dict["output"]
        total_mse = ((output - y) ** 2).sum() / len(test_x)
        ux_mse = ((output[:, 0, :, :] - test_y[:, 0, :, :]) ** 2).sum() / len(test_x)
        uy_mse = ((output[:, 1, :, :] - test_y[:, 1, :, :]) ** 2).sum() / len(test_x)
        p_mse = ((output[:, 2, :, :] - test_y[:, 2, :, :]) ** 2).sum() / len(test_x)
        return {
            "Total_MSE": total_mse,
            "Ux_MSE": ux_mse,
            "Uy_MSE": uy_mse,
            "p_MSE": p_mse,
        }

    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.FunctionalLoss(loss_expr),
        {"output": lambda out: out["output"]},
        {"MSE": ppsci.metric.FunctionalMetric(metric_expr)},
        name="mse_validator",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
        eval_during_train=True,
        eval_freq=50,
        validator=validator,
    )

    # train model
    solver.train()

    # evaluate after finished training
    solver.eval()

    PLOT_DIR = os.path.join(OUTPUT_DIR, "visual")
    os.makedirs(PLOT_DIR, exist_ok=True)
    VISU_INDEX = 0

    # visualize prediction after finished training
    predict_and_save_plot(test_x, test_y, VISU_INDEX, solver, PLOT_DIR)

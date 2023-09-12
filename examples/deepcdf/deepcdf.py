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

import ppsci
from ppsci.utils import logger


def split_tensors(
    *tensors: List[np.array], ratio: float
) -> Tuple[List[np.array], List[np.array]]:
    """Split tensors to two parts.

    Args:
      tensors (List[np.array]): Non-empty tensor list.
      ratio (float): Split ratio. For example, tensor list A is split to A1 and A2. len(A1) / len(A) = ratio.
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


if __name__ == "__main__":
    ppsci.utils.misc.set_random_seed(42)

    DATASET_PATH = "./datasets/deepCDF/"
    OUTPUT_DIR = "./output_deepCDF/"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # initialize datasets
    with open(os.path.join(DATASET_PATH, "dataX.pkl"), "rb") as file:
        x = pickle.load(file)
    with open(os.path.join(DATASET_PATH, "dataY.pkl"), "rb") as file:
        y = pickle.load(file)

    # slipt dataset to train dataset and test datatset
    SLIPT_RATIO = 0.7
    train_dataset, test_dataset = split_tensors(x, y, ratio=SLIPT_RATIO)
    train_x, train_y = train_dataset[:]
    test_x, test_y = test_dataset[:]

    CHANNELS_WEIGHTS = np.reshape(
        np.sqrt(
            np.mean(
                np.transpose(y, (0, 2, 3, 1)).reshape((981 * 172 * 79, 3)) ** 2, axis=0
            )
        ),
        (1, -1, 1, 1),
    )

    # initialize parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    KERNEL_SIZE = 5
    FILTERS = [8, 16, 32, 32]
    BATCH_NORM = False
    WEIGHT_NORM = False
    EPOCHS = 1000
    LEARNING_RATE = 0.001
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

    # initialize Adam optimizer
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)(model)

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

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=EPOCHS,
    )

    solver.train()

    ############### evaluation after training ###############
    output_dict = solver.predict({"input": test_x}, return_numpy=True)
    out = output_dict["output"]

    Total_MSE = ((out - test_y) ** 2).sum() / len(test_x)
    Ux_MSE = ((out[:, 0, :, :] - test_y[:, 0, :, :]) ** 2).sum() / len(test_x)
    Uy_MSE = ((out[:, 1, :, :] - test_y[:, 1, :, :]) ** 2).sum() / len(test_x)
    p_MSE = ((out[:, 2, :, :] - test_y[:, 2, :, :]) ** 2).sum() / len(test_x)

    logger.info(
        f"Total MSE is {Total_MSE:.5f}, Ux MSE is {Ux_MSE:.5f}, Uy MSE is {Uy_MSE:.5f}, p MSE is {p_MSE:.5f}"
    )

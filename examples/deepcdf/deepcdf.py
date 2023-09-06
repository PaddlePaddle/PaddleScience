import os
import pickle

import numpy as np

import ppsci
from ppsci.utils import logger


def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[: int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio) :])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2


if __name__ == "__main__":
    ppsci.utils.misc.set_random_seed(42)

    # DATASET_PATH = "./datasets/deepCDF/"
    DATASET_PATH = "/home/my/Share/"
    OUTPUT_DIR = "./output_deepCDF/"

    # initialize datasets
    with open(os.path.join(DATASET_PATH, "dataX.pkl"), "rb") as file:
        x = pickle.load(file)
    with open(os.path.join(DATASET_PATH, "dataY.pkl"), "rb") as file:
        y = pickle.load(file)

    train_dataset, test_dataset = split_tensors(x, y, ratio=float(0.7))
    train_x, train_y = train_dataset[:]
    test_x, test_y = test_dataset[:]

    channels_weights = np.reshape(
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
    KERNET_SIZE = 5
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
        filters=FILTERS,
        kernel_size=KERNET_SIZE,
        batch_norm=BATCH_NORM,
        weight_norm=WEIGHT_NORM,
    )

    # initialize Adam optimizer
    optimizer = ppsci.optimizer.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)(model)

    # define loss
    def loss_expr(output_dict, *args):
        output = output_dict["output"]
        y = args[0]["output"]
        lossu = ((output[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3])
        )
        lossv = ((output[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(
            (output.shape[0], 1, output.shape[2], output.shape[3])
        )
        lossp = (
            ((output[:, 2, :, :] - y[:, 2, :, :])).reshape(
                (output.shape[0], 1, output.shape[2], output.shape[3])
            )
        ).abs()
        loss = (lossu + lossv + lossp) / channels_weights
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
                "name": "DistributedBatchSampler",
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
        constraint=constraint,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer,
        epochs=EPOCHS,
    )

    solver.train()

    ############### evaluation after training ###############
    output_dict = solver.predict({"input": test_x})
    out = output_dict["output"]

    Total_MSE = ((out - test_y) ** 2).sum() / len(test_x)
    Ux_MSE = ((out[:, 0, :, :] - test_y[:, 0, :, :]) ** 2).sum() / len(test_x)
    Uy_MSE = ((out[:, 1, :, :] - test_y[:, 1, :, :]) ** 2).sum() / len(test_x)
    p_MSE = ((out[:, 2, :, :] - test_y[:, 2, :, :]) ** 2).sum() / len(test_x)

    logger.info(
        "Total MSE is {}, Ux MSE is {}, Uy MSE is {}, p MSE is {}".format(
            Total_MSE.detach().numpy()[0],
            Ux_MSE.detach().numpy()[0],
            Uy_MSE.detach().numpy()[0],
            p_MSE.detach().numpy()[0],
        )
    )

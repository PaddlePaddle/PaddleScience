import os
import pickle

import numpy as np
import paddle

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
    paddle.seed(999)

    x = pickle.load(open(os.path.join("/home/my/Share/", "dataX.pkl"), "rb"))
    y = pickle.load(open(os.path.join("/home/my/Share/", "dataY.pkl"), "rb"))
    train_dataset, test_dataset = split_tensors(x, y, ratio=float(0.7))

    train_x, train_y = train_dataset[:]
    test_x, test_y = test_dataset[:]
    # x = np.array(x)
    # x = x.transpose(0, 2, 3, 1)
    # print(x.shape)

    channels_weights = np.reshape(
        np.sqrt(
            np.mean(
                np.transpose(y, (0, 2, 3, 1)).reshape((981 * 172 * 79, 3)) ** 2, axis=0
            )
        ),
        (1, -1, 1, 1),
    )

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

    BATCH_SIZE = 64

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

    constraint = {sup_constraint.name: sup_constraint}

    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    KERNET_SIZE = 5
    FILTERS = [8, 16, 32, 32]
    BATCH_NORM = False
    WEIGHT_NORM = False
    EPOCHS = 1000
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.005

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

    optimizer = ppsci.optimizer.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)(model)
    solver = ppsci.solver.Solver(
        model, constraint=constraint, output_dir=".", optimizer=optimizer, epochs=EPOCHS
    )
    # solver = ppsci.solver.Solver(
    #     model,
    #     constraint=constraint,
    #     output_dir=".",
    #     optimizer=optimizer,
    #     epochs=EPOCHS,
    #     checkpoint_path="./checkpoints/latest",
    # )
    solver.train()
    output_dict = solver.predict({"input": test_x})
    out = output_dict["output"]
    Total_MSE = ((out - test_y) ** 2).sum() / len(test_x)
    Ux_MSE = ((out[:, 0, :, :] - test_y[:, 0, :, :]) ** 2).sum() / len(test_x)
    Uy_MSE = ((out[:, 1, :, :] - test_y[:, 1, :, :]) ** 2).sum() / len(test_x)
    p_MSE = ((out[:, 2, :, :] - test_y[:, 2, :, :]) ** 2).sum() / len(test_x)
    logger.message(
        "Total MSE is {}, Ux MSE is {}, Uy MSE is {}, p MSE is {}".format(
            Total_MSE.detach().numpy()[0],
            Ux_MSE.detach().numpy()[0],
            Uy_MSE.detach().numpy()[0],
            p_MSE.detach().numpy()[0],
        )
    )

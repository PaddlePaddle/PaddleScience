import paddle
from data_utils import augmentation
from paddle import nn
from prepare_datasets import generate_train_test
from sampler_utils import poisson_sampler
from TopOptModel import TopOptNN

import ppsci
from ppsci.utils import logger

if __name__ == "__main__":
    ### CASE 3: poisson(30)
    SIMP_stop_point_sampler = poisson_sampler(30)

    OUTPUT_DIR = "./Outputs/Poisson30_vol_coeff1/"
    DATA_PATH = "./Dataset/PreparedData/top_dataset.h5"

    BATCH_SIZE = 64
    N_SAMPLE = 10000
    TRAIN_TEST_RATIO = 1.0

    NUM_EPOCHS = 30
    VOL_COEFF = 1
    LEARNING_RATE = 0.001 / (1 + NUM_EPOCHS // 15)
    ITERS_PER_EPOCH = int(N_SAMPLE * TRAIN_TEST_RATIO / BATCH_SIZE)
    NUM_PARAMS = 192113  # the number given in paper

    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # generate_train_test dataset and dataloader
    X_train, Y_train = generate_train_test(DATA_PATH, TRAIN_TEST_RATIO, N_SAMPLE)

    # cnn model
    model = TopOptNN(channel_sampler=SIMP_stop_point_sampler)
    assert model.num_params == NUM_PARAMS

    # optimizer
    optimizer = ppsci.optimizer.Adam(learning_rate=LEARNING_RATE, epsilon=1e-07)(
        model
    )  # epsilon = 1e-07 is the default in tf

    # loss
    def loss_expr(output_dict, label_dict, weight_dict=None):
        y = label_dict["output"].reshape((-1, 1))
        y_pred = output_dict["output"].reshape((-1, 1))
        conf_loss = paddle.mean(
            nn.functional.log_loss(y_pred, y, epsilon=1e-7)
        )  # epsilon = 1e-07 is the default in tf
        vol_loss = paddle.square(paddle.mean(y - y_pred))
        return conf_loss + VOL_COEFF * vol_loss

    # constraints
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {"input": X_train},
                "label": {"output": Y_train},
            },
            "batch_size": BATCH_SIZE,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": False,
                "shuffle": True,
            },
            "transforms": (
                {
                    "FunctionalTransform": {
                        "transform_func": augmentation,
                    },
                },
            ),
        },
        ppsci.loss.FunctionalLoss(loss_expr),
        name="sup_constraint",
    )
    constraint = {sup_constraint.name: sup_constraint}

    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        epochs=NUM_EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
    )

    solver.train()

import paddle
from data_utils import NewNamedArrayDataset
from data_utils import NewSupConstraint
from data_utils import augmentation
from data_utils import batch_transform_wrapper
from paddle import nn
from paddle.io.dataloader import BatchSampler
from prepare_datasets import generate_train_test
from sampler_utils import poisson_sampler
from TopOptModel import TopOptNN

import ppsci
from ppsci.utils import logger

if __name__ == "__main__":
    ### CASE 1: poisson(5)
    SIMP_stop_point_sampler = poisson_sampler(5)

    OUTPUT_DIR = "./Outputs/Poisson5_vol_coeff1/"
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
    trainset = NewNamedArrayDataset(
        {"input": X_train}, {"output": Y_train}, transforms=augmentation
    )
    train_loader = paddle.io.DataLoader(
        dataset=trainset,
        batch_sampler=BatchSampler(
            dataset=trainset,
            sampler=None,
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=False,
        ),
        collate_fn=batch_transform_wrapper(SIMP_stop_point_sampler),
    )

    # cnn model
    model = TopOptNN()
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
    sup_constraint = NewSupConstraint(
        dataset=trainset,
        data_loader=train_loader,
        loss=ppsci.loss.FunctionalLoss(loss_expr),
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

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

import paddle
from data_utils import augmentation
from eval import evaluation_and_plot
from paddle import nn
from prepare_datasets import generate_train_test
from sampler_utils import generate_sampler
from TopOptModel import TopOptNN

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    ppsci.utils.misc.set_random_seed(42)
    DATA_PATH = "./Dataset/PreparedData/top_dataset.h5"

    # 4 training cases parameters
    CASE_PARAM = [("Poisson", 5), ("Poisson", 10), ("Poisson", 30), ("Uniform", None)]

    # specify model parameters
    IN_CHANNELS = 2
    OUT_CHANNELS = 1
    KERNEL_SIZE = 3
    FILTERS = (16, 32, 64)
    LAYERS = 2
    ACTIVATION = nn.ReLU

    # specify training parameters
    N_SAMPLE = 10000
    TRAIN_TEST_RATIO = (
        1.0  # use 10000 original data with different channels for training
    )
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001 / (1 + NUM_EPOCHS // 15)
    ITERS_PER_EPOCH = int(N_SAMPLE * TRAIN_TEST_RATIO / BATCH_SIZE)
    VOL_COEFF = 1  # coefficient for volume fraction constraint in the loss - beta in equation (3) in paper
    NUM_PARAMS = 192113  # the number of parameters in Unet specified in paper, used to verify the parameter number

    # generate training dataset
    X_train, Y_train = generate_train_test(DATA_PATH, TRAIN_TEST_RATIO, N_SAMPLE)

    # define loss
    def loss_expr(output_dict, label_dict, weight_dict=None):
        y = label_dict["output"].reshape((-1, 1))
        y_pred = output_dict["output"].reshape((-1, 1))
        conf_loss = paddle.mean(nn.functional.log_loss(y_pred, y, epsilon=1e-7))
        vol_loss = paddle.square(paddle.mean(y - y_pred))
        return conf_loss + VOL_COEFF * vol_loss

    # set constraints
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

    # train models for 4 cases
    for sampler_key, num in CASE_PARAM:

        # initialize SIMP iteration stop time sampler
        SIMP_stop_point_sampler = generate_sampler(sampler_key, num)

        # initialize logger
        OUTPUT_DIR = "Output_TopOpt" if args.output_dir is None else args.output_dir
        OUTPUT_DIR = os.path.join(
            OUTPUT_DIR,
            "".join(
                [
                    sampler_key,
                    str(num) if num is not None else "",
                    "_vol_coeff",
                    str(VOL_COEFF),
                ]
            ),
        )
        logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

        # set model
        model = TopOptNN(
            in_channel=IN_CHANNELS,
            out_channel=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            filters=FILTERS,
            layers=LAYERS,
            channel_sampler=SIMP_stop_point_sampler,
            activation=ACTIVATION,
        )
        assert model.num_params == NUM_PARAMS

        # set optimizer
        optimizer = ppsci.optimizer.Adam(learning_rate=LEARNING_RATE, epsilon=1e-07)(
            model
        )

        # initialize solver
        solver = ppsci.solver.Solver(
            model,
            constraint,
            OUTPUT_DIR,
            optimizer,
            epochs=NUM_EPOCHS,
            iters_per_epoch=ITERS_PER_EPOCH,
        )

        # train model
        solver.train()

    # evaluate 4 models
    evaluation_and_plot(sup_constraint.data_loader)

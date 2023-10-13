# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PhyCRNet for solving spatiotemporal PDEs
Reference: https://github.com/isds-neu/PhyCRNet/
"""
import functions
import numpy as np
import paddle
import scipy.io as scio

import ppsci
from ppsci.arch import phycrnet
from ppsci.utils import config
from ppsci.utils import logger


# transform
def transform_in(input):
    shape = input["initial_state_shape"][0]
    input_transformed = {
        "initial_state": input["initial_state"][0].reshape(shape.tolist()),
        "input": input["input"][0],
    }
    return input_transformed


def transform_out(input, out, model):
    # Stop the transform.
    model.enable_transform = False
    global dt, dx
    global num_time_batch

    loss_func = phycrnet.loss_generator(dt, dx)
    batch_loss = 0
    state_detached = []
    prev_output = []
    for time_batch_id in range(num_time_batch):
        # update the first input for each time batch
        if time_batch_id == 0:
            hidden_state = input["initial_state"]
            u0 = input["input"]
        else:
            hidden_state = state_detached
            u0 = prev_output[-2:-1].detach()  # second last output
            out = model({"initial_state": hidden_state, "input": u0})

        # output is a list
        output = out["outputs"]
        second_last_state = out["second_last_state"]

        # [t, c, height (Y), width (X)]
        output = paddle.concat(tuple(output), axis=0)

        # concatenate the initial state to the output for central diff
        output = paddle.concat((u0.cuda(), output), axis=0)

        # get loss
        loss = functions.compute_loss(output, loss_func)
        # loss.backward(retain_graph=True)
        batch_loss += loss

        # update the state and output for next batch
        prev_output = output
        state_detached = []
        for i in range(len(second_last_state)):
            (h, c) = second_last_state[i]
            state_detached.append((h.detach(), c.detach()))  # hidden state

    model.enable_transform = True
    return {"loss": batch_loss}


def tranform_output_val(input, out):
    global uv
    output = out["outputs"]
    input = input["input"]

    # shape: [t, c, h, w]
    output = paddle.concat(tuple(output), axis=0)
    output = paddle.concat((input.cuda(), output), axis=0)

    # Padding x and y axis due to periodic boundary condition
    output = paddle.concat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), axis=3)
    output = paddle.concat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), axis=2)

    # [t, c, h, w]
    truth = uv[0:1001, :, :, :]

    # [101, 2, 131, 131]
    truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # post-process
    ten_true = []
    ten_pred = []
    for i in range(0, 50):
        u_star, u_pred, v_star, v_pred = functions.post_process(
            output,
            truth,
            num=20 * i,
        )

        ten_true.append([u_star, v_star])
        ten_pred.append([u_pred, v_pred])

    # compute the error
    error = functions.frobenius_norm(
        np.array(ten_pred) - np.array(ten_true)
    ) / functions.frobenius_norm(np.array(ten_true))
    return {"loss": paddle.to_tensor([error])}


def train_loss_func(result_dict, *args) -> paddle.Tensor:
    return result_dict["loss"]


def val_loss_func(result_dict, *args) -> paddle.Tensor:
    return result_dict["loss"]


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(5)
    # set output directory
    OUTPUT_DIR = "./output_PhyCRNet" if not args.output_dir else args.output_dir
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")
    # set training hyper-parameters
    EPOCHS = 200 if not args.epochs else args.epochs

    # set initial states for convlstm
    num_convlstm = 1
    (h0, c0) = (paddle.randn((1, 128, 16, 16)), paddle.randn((1, 128, 16, 16)))
    initial_state = []
    for i in range(num_convlstm):
        initial_state.append((h0, c0))

    global num_time_batch
    global uv, dt, dx
    # grid parameters
    time_steps = 1001
    dt = 0.002
    dx = 1.0 / 128

    time_batch_size = 1000
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    model = ppsci.arch.PhyCRNet(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps,
        effective_step=effective_step,
    )

    def transform_out_wrap(_in, _out):
        return transform_out(_in, _out, model)

    model.register_input_transform(transform_in)
    model.register_output_transform(transform_out_wrap)

    # use burgers_data.py to generate data
    data_file = "./output/burgers_1501x2x128x128.mat"
    data = scio.loadmat(data_file)
    uv = data["uv"]  # [t,c,h,w]

    # initial condition
    uv0 = uv[0:1, ...]
    input = paddle.to_tensor(uv0, dtype=paddle.get_default_dtype())

    initial_state = paddle.to_tensor(initial_state)
    dataset_obj = functions.Dataset(initial_state, input)
    (
        input_dict_train,
        label_dict_train,
        input_dict_val,
        label_dict_val,
    ) = dataset_obj.get(EPOCHS)

    sup_constraint_pde = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_train,
                "label": label_dict_train,
            },
        },
        ppsci.loss.FunctionalLoss(train_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        name="sup_train",
    )
    constraint_pde = {sup_constraint_pde.name: sup_constraint_pde}

    sup_validator_pde = ppsci.validate.SupervisedValidator(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": input_dict_val,
                "label": label_dict_val,
            },
        },
        ppsci.loss.FunctionalLoss(val_loss_func),
        {
            "loss": lambda out: out["loss"],
        },
        metric={"metric": ppsci.metric.FunctionalMetric(functions.metric_expr)},
        name="sup_valid",
    )
    validator_pde = {sup_validator_pde.name: sup_validator_pde}

    # initialize solver
    ITERS_PER_EPOCH = 1
    scheduler = ppsci.optimizer.lr_scheduler.Step(
        epochs=EPOCHS,
        iters_per_epoch=ITERS_PER_EPOCH,
        step_size=100,
        gamma=0.97,
        learning_rate=1e-4,
    )()
    optimizer = ppsci.optimizer.Adam(scheduler)(model)
    solver = ppsci.solver.Solver(
        model,
        constraint_pde,
        OUTPUT_DIR,
        optimizer,
        scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=50,
        validator=validator_pde,
        eval_with_no_grad=True,
    )

    # train model
    solver.train()
    # evaluate after finished training
    model.register_output_transform(tranform_output_val)
    solver.eval()

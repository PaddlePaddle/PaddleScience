# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""This *.py file is the solver class of PDSC"""
import os
import time

import numpy as np
import paddle
from paddle.distributed.fleet import auto
from paddle.incubate.optimizer.functional.bfgs import minimize_bfgs
from paddle.incubate.optimizer.functional.lbfgs import minimize_lbfgs
from visualdl import LogWriter

from .. import config
from .. import logging
from . import utils

__all__ = ["Solver"]


class DataSetStatic(paddle.io.Dataset):
    def __init__(self, nsamples, inputs_labels):
        self.inputs = inputs_labels
        self.nsamples = nsamples

    def __getitem__(self, idx):
        return self.inputs

    def __len__(self):
        return self.nsamples


class ModelStatic(paddle.nn.Layer):
    def __init__(self, pde, algo, ninputs, inputs_attr, nlabels, labels_attr):
        super(ModelStatic, self).__init__()
        self.pde = pde
        self.algo = algo
        self.ninputs = ninputs
        self.inputs_attr = inputs_attr
        self.nlabels = nlabels
        self.labels_attr = labels_attr

        self.algo.net.make_network()

    def forward(self, *inputs_labels):
        for input in inputs_labels:
            input.stop_gradient = False

        self.loss, self.outs, self.loss_details = self.algo.compute(
            None,
            *inputs_labels,
            ninputs=self.ninputs,
            inputs_attr=self.inputs_attr,
            nlabels=self.nlabels,
            labels_attr=self.labels_attr,
            pde=self.pde,
        )

        auto.fetch(self.loss_details[0], "eq_loss")
        auto.fetch(self.loss_details[1], "bc_loss")
        auto.fetch(self.loss_details[2], "ic_loss")
        auto.fetch(self.loss_details[3], "data_loss")

        return self.loss, self.outs  # TODO: add outs


def loss_func(x, y):

    # print("\n ********** loss_func done ****  \n")
    return x


class Solver(object):
    """
    Solver

    Parameters:
        pde(paddlescience.pde): The PDE used in the solver.
        algo(Algorithm): The algorithm used in the solver.
        opt(paddlescience.Optimizer, optional): The optimizer used in the solver.

    Example:
        >>> # 1. train
        >>> import paddlescience as psci
        >>> solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
        >>> solution = solver.solve()

        >>> # 2. predict
        >>> import paddlescience as psci
        >>> solver = psci.solver.Solver(pde=pde_disc, algo=algo)
        >>> solution = solver.predict()
    """

    # init
    def __init__(
        self,
        pde,
        algo,
        opt=None,
        data_driven=False,
        lr_scheduler=None,
        lr_update_method="epoch",
        train_dataloader=None,
        valid_dataloader=None,
    ):
        super(Solver, self).__init__()

        self.pde = pde
        self.algo = algo
        self.opt = opt
        self._dtype = config._dtype

        self.data_driven = data_driven
        self.lr_scheduler = lr_scheduler

        assert lr_update_method in [
            "epoch",
            "step",
        ], "invalid lr update method: {!r}".format(lr_update_method)
        self.lr_update_method = lr_update_method

        assert (train_dataloader is not None and valid_dataloader is not None) or (
            train_dataloader is None and valid_dataloader is None
        )
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        if paddle.in_dynamic_mode():
            if data_driven is False:
                self.__init_dynamic()
        else:
            if paddle.distributed.get_world_size() == 1:
                self.__init_static()
            else:
                self.__init_static_auto_dist()

    # solve (train)
    def solve(
        self,
        num_epoch=2,
        bs=None,
        checkpoint_freq=1000,
        checkpoint_path="./checkpoint/",
    ):
        if paddle.in_dynamic_mode():
            if self.data_driven is False:
                return self.__solve_dynamic(
                    num_epoch, bs, checkpoint_freq, checkpoint_path
                )
            else:
                return self.__solve_dynamic_data_driven(
                    num_epoch, bs, checkpoint_freq, checkpoint_path
                )
        else:
            if paddle.distributed.get_world_size() == 1:
                return self.__solve_static(
                    num_epoch, bs, checkpoint_freq, checkpoint_path
                )
            else:
                return self.__solve_static_auto_dist(num_epoch, bs, checkpoint_freq)

    # predict (infer)
    def predict(self):
        if paddle.in_dynamic_mode():
            return self.__predict_dynamic()
        else:
            if paddle.distributed.get_world_size() == 1:
                return self.__predict_static()
            else:
                return self.__predict_static_auto_dist()

    # init dynamic
    def __init_dynamic(self):
        # """
        # Train the network with respect to num_epoch.

        # Parameters:
        #     -num_epoch(int): Optional, default 1000. Number of epochs.
        #     -batch_size(int|None): Under develop. Optional, default None.
        #       How many sample points are used as a batch during training.
        #     -checkpoint_freq(int): Under develop. Optional, default 1000.
        #       How many epochs to store the training status once.

        # Return:
        #     solution(Callable):A python func functhion that takes
        #       a GeometryDiscrete as input and a numpy array as outputs.

        # Example:
        #     >>> import paddlescience as psci
        #     >>> solver = psci.solver.Solver(algo=algo, opt=opt)
        #     >>> solution = solver.solve(num_epoch=10000)
        #     >>> rslt = solution(geo)
        # """

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        self.inputs = inputs
        self.inputs_attr = inputs_attr

        if self.opt is not None:
            labels, labels_attr = self.algo.create_labels(self.pde)
            self.labels = labels
            self.labels_attr = labels_attr

    # solve static
    def __solve_dynamic(self, num_epoch, bs, checkpoint_freq, checkpoint_path):

        inputs = self.inputs
        inputs_attr = self.inputs_attr
        labels = self.labels
        labels_attr = self.labels_attr

        # number of inputs and labels
        ninputs = len(inputs)
        nlabels = len(labels)

        # convert inputs to tensor
        for i in range(ninputs):
            inputs[i] = paddle.to_tensor(
                inputs[i], dtype=self._dtype, stop_gradient=False
            )

        # convert label to tensor
        for i in range(nlabels):
            labels[i] = paddle.to_tensor(
                labels[i], dtype=self._dtype, stop_gradient=False
            )

        inputs_labels = inputs + labels  # tmp to one list

        print("Dynamic Graph is Currently in Use.")
        if config.visualdl_enabled() == True:
            writer_loss = LogWriter(logdir=checkpoint_path + "visualDL/loss")
            writer_eq_loss = LogWriter(logdir=checkpoint_path + "visualDL/eq_loss")
            writer_bc_loss = LogWriter(logdir=checkpoint_path + "visualDL/bc_loss")
            writer_ic_loss = LogWriter(logdir=checkpoint_path + "visualDL/ic_loss")
            writer_data_loss = LogWriter(logdir=checkpoint_path + "visualDL/data_loss")
        # Adam optimizer
        if isinstance(self.opt, paddle.optimizer.AdamW) or isinstance(
            self.opt, paddle.optimizer.Adam
        ):

            # record time
            timer = utils.Timer()
            if os.getenv("TIPC_TEST") == "ON":  # script will turn this env variable one
                tipc_test_mode = True
            else:
                tipc_test_mode = False

            if tipc_test_mode is True:  # tipc test
                # N个Step打印1条日志时，reader_cost为N个Step数据加载用时的平均值，全量训练，N=1
                N_print = 1
                Rn = np.zeros((N_print,))  # 每Step reader用时为：R1, R2,...Rn
                Tn = np.zeros((N_print,))  # 每Step训练用时：T1, T2,...Tn
                Sn = np.zeros((N_print,))  # 每Step 单卡BatchSize 为S1, S2,...Sn
                Sn[0] = 1  # Full batch training
                time_point = np.zeros((num_epoch + 1, 1))
                samples = 0  # samples代表上次打印到本次打印，新完成训练的样本数量
                for i in range(ninputs):
                    samples += inputs[i].shape[0]
                time_point[0] = time.perf_counter()
                reader_cost = 0  # modify it if loading data from file

            for epoch in range(num_epoch):

                # TODO: error out num_epoch==0
                # forward computation start, log the time for test_tipc
                time_start = time.perf_counter()
                loss, outs, loss_details = self.algo.compute(
                    None,
                    *inputs_labels,
                    ninputs=ninputs,
                    inputs_attr=inputs_attr,
                    nlabels=nlabels,
                    labels_attr=labels_attr,
                    pde=self.pde,
                )

                if tipc_test_mode is True:  # tipc test
                    time_end = time.perf_counter()
                    time_forward_cost = time_end - time_start
                    batch_cost = reader_cost + time_forward_cost
                    avg_reader_cost = np.sum(Rn) / N_print
                    Tn[0] = batch_cost
                    avg_batch_cost = avg_reader_cost + np.sum(Tn) / N_print
                    avg_samples = np.sum(Sn) / N_print
                    ips = samples / batch_cost

                loss.backward()
                self.opt.step()
                self.opt.clear_grad()

                print(
                    "epoch: " + str(epoch + 1),
                    " loss:",
                    float(loss),
                    " eq loss:",
                    float(loss_details[0]),
                    " bc loss:",
                    float(loss_details[1]),
                    " ic loss:",
                    float(loss_details[2]),
                    " data loss:",
                    float(loss_details[3]),
                )

                if tipc_test_mode is True:  # tipc test
                    print(
                        f"..., ... , loss: "
                        f"{float(loss)}, avg_reader_cost: "
                        f"{avg_reader_cost} sec, avg_batch_cost: "
                        f"{avg_batch_cost} sec, avg_samples: "
                        f"{avg_samples}, ips: "
                        f"{ips} samples/s"
                    )

                # write loss for visual DL
                if config.visualdl_enabled() == True:
                    writer_loss.add_scalar(tag="loss", step=epoch, value=float(loss))
                    writer_eq_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(loss_details[0])
                    )
                    writer_bc_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(loss_details[1])
                    )
                    writer_ic_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(loss_details[2])
                    )
                    writer_data_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(loss_details[3])
                    )

                if (epoch + 1) % checkpoint_freq == 0:
                    paddle.save(
                        self.algo.net.state_dict(),
                        checkpoint_path
                        + "dynamic_net_params_"
                        + str(epoch + 1)
                        + ".pdparams",
                    )
                    paddle.save(
                        self.opt.state_dict(),
                        checkpoint_path
                        + "dynamic_opt_params_"
                        + str(epoch + 1)
                        + ".pdopt",
                    )

            # print time
            timer.end()
            timer.print()

            for i in range(len(outs)):
                outs[i] = outs[i].numpy()

            return outs
        # L-bfgs optimizer
        elif self.opt is minimize_lbfgs or self.opt is minimize_bfgs:

            def _f(x):
                self.algo.net.reconstruct(x)
                loss, self.outs, self.loss_details = self.algo.compute(
                    None,
                    *inputs_labels,
                    ninputs=ninputs,
                    inputs_attr=inputs_attr,
                    nlabels=nlabels,
                    labels_attr=labels_attr,
                    pde=self.pde,
                )
                return loss

            x0 = self.algo.net.flatten_params()

            for epoch in range(num_epoch):
                results = self.opt(
                    _f,
                    x0,
                    initial_inverse_hessian_estimate=None,
                    line_search_fn="strong_wolfe",
                    dtype=config._dtype,
                )
                x0 = results[2]

                print(
                    "epoch: " + str(epoch + 1),
                    " loss:",
                    float(results[3]),
                    " eq loss:",
                    float(self.loss_details[0]),
                    " bc loss:",
                    float(self.loss_details[1]),
                    " ic loss:",
                    float(self.loss_details[2]),
                    " data loss:",
                    float(self.loss_details[3]),
                )

                # write loss for visual DL
                if config.visualdl_enabled() == True:
                    writer_loss.add_scalar(
                        tag="loss", step=epoch, value=float(results[3])
                    )
                    writer_eq_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(self.loss_details[0])
                    )
                    writer_bc_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(self.loss_details[1])
                    )
                    writer_ic_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(self.loss_details[2])
                    )
                    writer_data_loss.add_scalar(
                        tag="detail_loss", step=epoch, value=float(self.loss_details[3])
                    )

                if (epoch + 1) % checkpoint_freq == 0:
                    paddle.save(
                        self.algo.net.state_dict(),
                        checkpoint_path
                        + "dynamic_net_params_"
                        + str(epoch + 1)
                        + ".pdparams",
                    )

            self.algo.net.reconstruct(x0)

            for i in range(len(self.outs)):
                self.outs[i] = self.outs[i].numpy()

            return self.outs
        else:
            print(
                "Please specify the optimizer, now only the adam, "
                "lbfgs and bfgs optimizers are supported."
            )
            exit()

        # close writer in visual DL
        if config.visualdl_enabled() == True:
            writer_loss.close()
            writer_eq_loss.close()
            writer_bc_loss.close()
            writer_ic_loss.close()
            writer_data_loss.close()

    def __solve_dynamic_data_driven(
        self, num_epoch, bs, checkpoint_freq, checkpoint_path
    ):

        logger = logging.get_logger(log_file=os.path.join(checkpoint_path, "train.log"))
        if config.visualdl_enabled() == True:
            vdl_writer = LogWriter(
                logdir=os.path.join(checkpoint_path, "visualDL/loss")
            )

        # for train
        for epoch in range(num_epoch):
            loss_total = 0
            total_batch = len(self.train_dataloader)
            for idx, batch in enumerate(self.train_dataloader):

                losses = self.algo.compute(**batch)
                loss = losses["loss"]
                loss_total += loss
                cur_lr = self.opt.get_lr()
                if idx % 10 == 0 or idx + 1 == total_batch:
                    logger.info(
                        "epoch: [{}/{}], iter: [{}/{}], lr: {:.5f}, loss {:.5f}".format(
                            epoch, num_epoch, idx, total_batch, cur_lr, loss.item()
                        )
                    )
                step = epoch * total_batch + idx
                if config.visualdl_enabled() == True:
                    vdl_writer.add_scalar("loss", loss.item(), step)

                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
                if self.lr_update_method == "step":
                    self.lr_scheduler.step((epoch + 1) + float(idx) / total_batch)

            if self.lr_update_method == "epoch":
                self.lr_scheduler.step()
            logger.info(
                "epoch: [{}/{}], lr: {:.5f}, avg_loss {:.5f}".format(
                    epoch, num_epoch, cur_lr, loss_total.item() / total_batch
                )
            )

            if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epoch - 1:
                # for eval
                total_batch = len(self.valid_dataloader)
                loss_total = 0
                for idx, batch in enumerate(self.valid_dataloader):
                    batch["visu_dir"] = os.path.join(
                        checkpoint_path, "visu/{}".format(epoch + 1)
                    )
                    losses = self.algo.eval(**batch)
                    loss_total = loss_total + losses["loss"]
                    if idx % 10 == 0 or idx + 1 == total_batch:
                        logger.info(
                            "eval iter: [{}/{}], loss {:.5f}".format(
                                idx, total_batch, float(loss_total / (idx + 1))
                            )
                        )
                avg_loss = loss_total.item() / total_batch
                logger.info("eval avg_loss {:.5f}".format(avg_loss))

                paddle.save(
                    self.algo.net.state_dict(),
                    checkpoint_path
                    + "dynamic_net_params_"
                    + str(epoch + 1)
                    + ".pdparams",
                )
                paddle.save(
                    self.opt.state_dict(),
                    checkpoint_path + "dynamic_opt_params_" + str(epoch + 1) + ".pdopt",
                )

        if config.visualdl_enabled() == True:
            vdl_writer.close()

    # predict dynamic
    def __predict_dynamic(self):
        # create inputs
        inputs, inputs_attr = self.algo.create_inputs(self.pde)

        # convert inputs to tensor
        for i in range(len(inputs)):
            inputs[i] = paddle.to_tensor(
                inputs[i], dtype=self._dtype, stop_gradient=False
            )

        outs = self.algo.compute_forward(None, *inputs)

        for i in range(len(outs)):
            outs[i] = outs[i].numpy()

        return outs

    # init static
    def __init_static(self):
        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        self.inputs = inputs
        self.inputs_attr = inputs_attr

        if self.opt is not None:
            if config.prim_enabled() and self.pde.geometry.user is not None:
                labels, labels_attr = self.algo.create_labels(
                    self.pde,
                    interior_shape=len(self.pde.geometry.interior),
                    supervised_shape=len(self.pde.geometry.user),
                )
            else:
                labels, labels_attr = self.algo.create_labels(self.pde)
            self.labels = labels
            self.labels_attr = labels_attr

        place = paddle.CUDAPlace(0)
        self.exe = paddle.static.Executor(place)

        if self.opt is not None:
            # number of inputs and labels
            ninputs = len(self.inputs)
            nlabels = len(self.labels)

            inputs_labels = list()

            self.train_program = paddle.static.Program()
            self.startup_program = paddle.static.Program()

            # construct train program
            with paddle.static.program_guard(self.train_program, self.startup_program):

                # dynamic mode: make network in net's constructor
                # static  mode: make network here
                self.algo.net.make_network()

                # inputs
                for i in range(len(self.inputs)):
                    # inputs
                    input = paddle.static.data(
                        name="input" + str(i),
                        shape=self.inputs[i].shape,
                        dtype=self._dtype,
                    )
                    input.stop_gradient = False
                    inputs_labels.append(input)

                for i in range(len(self.labels)):
                    # labels
                    label = paddle.static.data(
                        name="label" + str(i),
                        shape=self.labels[i].shape,
                        dtype=self._dtype,
                    )
                    label.stop_gradient = False
                    inputs_labels.append(label)

                self.loss, self.outs, self.loss_details = self.algo.compute(
                    None,
                    *inputs_labels,
                    ninputs=ninputs,
                    inputs_attr=self.inputs_attr,
                    nlabels=nlabels,
                    labels_attr=self.labels_attr,
                    pde=self.pde,
                )

                if self.opt is minimize_lbfgs or self.opt is minimize_bfgs:
                    assert (
                        paddle.in_dynamic_mode()
                    ), "The lbfgs and bfgs optimizer is only supported in dynamic graph"
                self.opt.minimize(self.loss)

                # new ad
                if config.prim_enabled() and not config.cinn_enabled():
                    config.prim2orig()

            # startup program
            self.exe.run(self.startup_program)

    # solve static
    def __solve_static(self, num_epoch, bs, checkpoint_freq, checkpoint_path):
        inputs = self.inputs
        labels = self.labels

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds["input" + str(i)] = inputs[i]

        # feeds labels
        for i in range(len(labels)):
            feeds["label" + str(i)] = labels[i]

        # fetch loss and net's outputs
        fetches = [self.loss.name]
        for out in self.outs:
            fetches.append(out.name)
        # fetch loss_details' outputs
        for loss_detail in self.loss_details:
            fetches.append(loss_detail.name)

        # load model
        if self.algo.net.params_path is not None:
            state_dict = paddle.load(self.algo.net.params_path)
            self.train_program.set_state_dict(state_dict)

        # main loop
        print("Static Graph is Currently in Use.")
        if config.prim_enabled():
            print("Optimized AD is Currently in Use")

        if config.visualdl_enabled() == True:
            writer_loss = LogWriter(logdir=checkpoint_path + "visualDL/loss")
            writer_eq_loss = LogWriter(logdir=checkpoint_path + "visualDL/eq_loss")
            writer_bc_loss = LogWriter(logdir=checkpoint_path + "visualDL/bc_loss")
            writer_ic_loss = LogWriter(logdir=checkpoint_path + "visualDL/ic_loss")
            writer_data_loss = LogWriter(logdir=checkpoint_path + "visualDL/data_loss")

        if config.cinn_enabled():
            begin = time.time()
            print("CINN is currently used.")
            compiled_program = utils.cinn_compile(
                self.train_program, self.loss.name, fetches
            )
        else:
            compiled_program = self.train_program

        # record time
        timer = utils.Timer()

        for epoch in range(num_epoch):
            rslt = self.exe.run(compiled_program, feed=feeds, fetch_list=fetches)
            print(
                "epoch: " + str(epoch + 1),
                "loss: ",
                rslt[0],
                " eq loss:",
                rslt[-4],
                " bc loss:",
                rslt[-3],
                " ic loss:",
                rslt[-2],
                " data loss:",
                rslt[-1],
            )

            # write loss for visual DL
            if config.visualdl_enabled() == True:
                writer_loss.add_scalar(tag="loss", step=epoch, value=rslt[0])
                writer_eq_loss.add_scalar(tag="detail_loss", step=epoch, value=rslt[-4])
                writer_bc_loss.add_scalar(tag="detail_loss", step=epoch, value=rslt[-3])
                writer_ic_loss.add_scalar(tag="detail_loss", step=epoch, value=rslt[-2])
                writer_data_loss.add_scalar(
                    tag="detail_loss", step=epoch, value=rslt[-1]
                )

            if (epoch + 1) % checkpoint_freq == 0:
                paddle.save(
                    self.train_program.state_dict(),
                    checkpoint_path
                    + "static_model_params_"
                    + str(epoch + 1)
                    + ".pdparams",
                )

            if config.cinn_enabled():
                if epoch == 0:
                    first_step_cost = time.time() - begin
                elif epoch == 9:
                    paddle.device.cuda.synchronize()
                    begin = time.time()
                elif epoch == num_epoch - 1:
                    paddle.device.cuda.synchronize()
                    end = time.time()
                    print("First step cost {} s".format(first_step_cost))
                    print(
                        "{} epoch(10~{}) cost {} s".format(
                            num_epoch - 10, num_epoch, end - begin
                        )
                    )
        # print time
        timer.end()
        timer.print()

        # close writer in visual DL
        if config.visualdl_enabled() == True:
            writer_loss.close()
            writer_eq_loss.close()
            writer_bc_loss.close()
            writer_ic_loss.close()
            writer_data_loss.close()

        return rslt[1:-4]

    # predict static
    def __predict_static(self):
        self.startup_program = paddle.static.Program()
        self.predict_program = paddle.static.Program()

        # construct predict program
        with paddle.static.program_guard(self.predict_program, self.startup_program):
            with paddle.utils.unique_name.guard():

                self.algo.net.make_network()
                ins = list()
                for i in range(len(self.inputs)):
                    ishape = list(self.inputs[i].shape)
                    ishape[0] = -1
                    input = paddle.static.data(
                        name="input" + str(i), shape=ishape, dtype=self._dtype
                    )
                    input.stop_gradient = False
                    ins.append(input)

                self.outs_predict = self.algo.compute_forward(None, *ins)

        # startup program
        self.exe.run(self.startup_program)

        # feeds inputs
        feeds = dict()
        for i in range(len(self.inputs)):
            feeds["input" + str(i)] = self.inputs[i]

        # fetch outputs
        fetches = list()
        for out in self.outs_predict:
            fetches.append(out.name)

        # load model
        if self.algo.net.params_path is not None:
            state_dict = paddle.load(self.algo.net.params_path)
            self.predict_program.set_state_dict(state_dict)
        else:
            assert 0, "Please specify the path and name of the static model."

        # run
        rslt = self.exe.run(self.predict_program, feed=feeds, fetch_list=fetches)

        return rslt

    # init in static mode with auto dist
    def __init_static_auto_dist(self):

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        self.inputs = inputs
        self.inputs_attr = inputs_attr
        self.labels = labels
        self.labels_attr = labels_attr

        # number of inputs and labels
        ninputs = len(inputs)
        nlabels = len(labels)

        # strategy
        dist_strategy = auto.Strategy()
        dist_strategy.auto_mode = "semi"
        dist_strategy.gradient_scale = False

        self.model = ModelStatic(
            self.pde, self.algo, ninputs, inputs_attr, nlabels, labels_attr
        )

        # engine
        self.engine = auto.Engine(
            self.model, loss_func, self.opt, strategy=dist_strategy
        )

    # solve in static mode with auto dist
    def __solve_static_auto_dist(self, num_epoch, bs, checkpoint_freq):

        # inputs and its attributes
        inputs = self.inputs
        inputs_attr = self.inputs_attr
        labels = self.labels
        labels_attr = self.labels_attr

        # dataset
        train_dataset = DataSetStatic(num_epoch, inputs + labels)

        timer = utils.Timer()

        # train
        self.engine.fit(train_dataset, len(inputs + labels), batch_size=None)

        # predict
        self.predict_auto_dist_program = paddle.fluid.Program()
        with paddle.static.program_guard(self.predict_auto_dist_program):
            with paddle.utils.unique_name.guard():

                self.algo.net.make_network()
                ins = list()
                for i in range(len(inputs)):
                    ishape = list(inputs[i].shape)
                    ishape[0] = -1
                    input = paddle.static.data(
                        name="input" + str(i), shape=ishape, dtype=self._dtype
                    )
                    input.stop_gradient = False
                    ins.append(input)

                self.outs_predict = self.algo.compute_forward(None, *ins)

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds["input" + str(i)] = inputs[i]

        # fetch_list
        fetches = []
        for out in self.outs_predict:
            fetches.append(out.name)

        rslt = self.engine._executor.run(
            self.predict_auto_dist_program, feed=feeds, fetch_list=fetches
        )

        timer.end()
        timer.print()

        return rslt

    # predict static auto-dist
    def __predict_static_auto_dist(self):

        # create inputs and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds["input" + str(i)] = inputs[i]

        # fetch_list
        fetches = []
        for out in self.outs_predict:
            fetches.append(out.name)

        rslt = self.engine._executor.run(
            self.predict_auto_dist_program, feed=feeds, fetch_list=fetches
        )
        return rslt

    def feed_data_interior_cur(self, data):
        self.labels = self.algo.feed_data_interior_cur(
            self.labels, self.labels_attr, data
        )

    def feed_data_user_cur(self, data):
        self.labels = self.algo.feed_data_user_cur(self.labels, self.labels_attr, data)

    def feed_data_user_next(self, data):
        self.labels = self.algo.feed_data_user_next(self.labels, self.labels_attr, data)

    def feed_data_user(self, data):
        self.feed_data_user_next(data)

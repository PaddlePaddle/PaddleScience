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
import numpy as np
import paddle
from paddle.static import InputSpec
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.engine import Engine
from .. import config

__all__ = ["Solver"]


class DataSetStatic:
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

        self.algo.net.make_network_static()

    def forward(self, *inputs_labels):

        loss, outs = self.algo.compute(
            *inputs_labels,
            ninputs=self.ninputs,
            inputs_attr=self.inputs_attr,
            nlabels=self.nlabels,
            labels_attr=self.labels_attr,
            pde=self.pde)

        # print("\n ********** compute done ****  \n")

        return loss, outs  # TODO: add outs


def loss_func(x, y):

    # print("\n ********** loss_func done ****  \n")
    return x


class Solver(object):
    """
    Solver
 
    Parameters:
        algo(AlgorithmBase): The algorithm used in the solver.
        opt(paddle.Optimizer): The optimizer used in the solver.

    Example:
        >>> import paddlescience as psci
        >>> solver = psci.solver.Solver(algo=algo, opt=opt)
    """

    def __init__(self, pde, algo, opt):
        super(Solver, self).__init__()

        self.pde = pde
        self.algo = algo
        self.opt = opt
        self._dtype = config._dtype

        if paddle.in_dynamic_mode():
            self.__init_dynamic()
        else:
            if paddle.distributed.get_world_size() == 1:
                self.__init_static()
            else:
                pass

    def solve(self, num_epoch=2, bs=None, checkpoint_freq=1000):

        if paddle.in_dynamic_mode():
            return self.__solve_dynamic(num_epoch, bs, checkpoint_freq)
        else:
            if paddle.distributed.get_world_size() == 1:
                return self.__solve_static(num_epoch, bs, checkpoint_freq)
            else:
                return self.__solve_static_auto_dist(num_epoch, bs,
                                                     checkpoint_freq)

    def feed_data_n(self, data_n):
        self.labels = self.algo.feed_labels_data_n(self.labels,
                                                   self.labels_attr, data_n)

    def feed_data(self, data):
        self.labels = self.algo.feed_labels_data(self.labels, self.labels_attr,
                                                 data)

    # solve in dynamic mode
    def __init_dynamic(self):
        """
        Train the network with respect to num_epoch.
 
        Parameters:
            num_epoch(int): Optional, default 1000. Number of epochs.
            batch_size(int|None): Under develop. Optional, default None. How many sample points are used as a batch during training.
            checkpoint_freq(int): Under develop. Optional, default 1000. How many epochs to store the training status once.

        Return:
            solution(Callable): A python func functhion that takes a GeometryDiscrete as input and a numpy array as outputs.

        Example:
            >>> import paddlescience as psci
            >>> solver = psci.solver.Solver(algo=algo, opt=opt)
            >>> solution = solver.solve(num_epoch=10000)
            >>> rslt = solution(geo)
        """

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        self.inputs = inputs
        self.inputs_attr = inputs_attr
        self.labels = labels
        self.labels_attr = labels_attr

    def __solve_dynamic(self, num_epoch, bs, checkpoint_freq):

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
                inputs[i], dtype=self._dtype, stop_gradient=False)

        # convert label to tensor
        for i in range(nlabels):
            labels[i] = paddle.to_tensor(
                labels[i], dtype=self._dtype, stop_gradient=False)

        inputs_labels = inputs + labels  # tmp to one list

        for epoch in range(num_epoch):

            # TODO: error out num_epoch==0

            loss, outs = self.algo.compute(
                *inputs_labels,
                ninputs=ninputs,
                inputs_attr=inputs_attr,
                nlabels=nlabels,
                labels_attr=labels_attr,
                pde=self.pde)

            loss.backward()
            self.opt.step()
            self.opt.clear_grad()

            print("dynamic epoch: " + str(epoch + 1), "    loss:",
                  loss.numpy()[0])

        for i in range(len(outs)):
            outs[i] = outs[i].numpy()

        return outs
        # def solution_fn(geo):
        #     if geo.time_dependent == False:
        #         if not isinputstance(geo.space_domain, paddle.Tensor):
        #             geo.set_batch_size(geo.get_domain_size())
        #             geo.to_tensor()
        #         return self.algo.net.nn_func(geo.space_domain).numpy()
        #     else:
        #         if not isinputstance(geo.domain, paddle.Tensor):
        #             geo.set_batch_size(geo.get_domain_size())
        #             geo.to_tensor()
        #         return self.algo.net.nn_func(geo.domain).numpy()

        # return solution_fn

    # solver in static mode

    def __init_static(self):

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        self.inputs = inputs
        self.inputs_attr = inputs_attr
        self.labels = labels
        self.labels_attr = labels_attr

        # number of inputs and labels
        ninputs = len(self.inputs)
        nlabels = len(self.labels)

        place = paddle.CUDAPlace(0)
        self.exe = paddle.static.Executor(place)

        inputs_labels = list()

        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()

        # construct program
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):

            # dynamic mode: make network in net's constructor
            # static  mode: make network here 
            self.algo.net.make_network_static()

            # inputs
            for i in range(len(inputs)):
                #inputs
                input = paddle.static.data(
                    name='input' + str(i),
                    shape=inputs[i].shape,
                    dtype=self._dtype)
                input.stop_gradient = False
                inputs_labels.append(input)

            for i in range(len(labels)):
                #labels
                label = paddle.static.data(
                    name='label' + str(i),
                    shape=labels[i].shape,
                    dtype=self._dtype)
                label.stop_gradient = False
                inputs_labels.append(label)

            self.loss, self.outs = self.algo.compute(
                *inputs_labels,
                ninputs=ninputs,
                inputs_attr=inputs_attr,
                nlabels=nlabels,
                labels_attr=labels_attr,
                pde=self.pde)

            self.opt.minimize(self.loss)

        # start up program
        self.exe.run(self.startup_program)

    def __solve_static(self, num_epoch, bs, checkpoint_freq):

        inputs = self.inputs
        inputs_attr = self.inputs_attr
        labels = self.labels
        labels_attr = self.labels_attr

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds['input' + str(i)] = inputs[i]

        # feeds labels
        for i in range(len(labels)):
            feeds['label' + str(i)] = labels[i]

        # fetch loss and net's outputs
        fetches = [self.loss.name]
        for out in self.outs:
            fetches.append(out.name)

        # main loop
        for epoch in range(num_epoch):
            rslt = self.exe.run(self.main_program,
                                feed=feeds,
                                fetch_list=fetches)
            print("static epoch: " + str(epoch + 1), "loss: ", rslt[0])

        return rslt[1:]

    # def __solve_static_dist(self, num_epoch, bs, checkpoint_freq):

    #     # init dist environment
    #     strategy = fleet.DistributedStrategy()
    #     fleet.init(is_collective=True, strategy=strategy)

    #     inputs, inputs_attr = self.algo.create_inputs(self.pde)

    #     place = paddle.CUDAPlace(0)
    #     exe = paddle.static.Executor(place)

    #     # dist optimizer
    #     opt_dist = fleet.distributed_optimizer(self.opt)

    #     inputs = list()
    #     feeds = dict()

    #     main_program = paddle.static.Program()
    #     startup_program = paddle.static.Program()

    #     # construct program
    #     with paddle.static.program_guard(main_program, startup_program):

    #         self.algo.net.make_network_static()

    #         for i in range(len(inputs)):
    #             # inputs
    #             input = paddle.static.data(
    #                 name='input' + str(i),
    #                 shape=inputs[i].shape,
    #                 dtype=self._dtype)
    #             input.stop_gradient = False
    #             inputs.append(input)

    #             # feeds
    #             feeds['input' + str(i)] = inputs[i]

    #         loss, outs = self.algo.compute(
    #             *inputs, inputs_attr=inputs_attr, pde=self.pde)

    #         opt_dist.minimize(loss)

    #     # fetch loss and net's output
    #     fetches = [loss.name]
    #     for out in outs:
    #         fetches.append(out.name)

    #     # start up program
    #     exe.run(startup_program)

    #     # main loop
    #     for epoch in range(num_epoch):
    #         rslt = exe.run(main_program, feed=feeds, fetch_list=fetches)
    #         print("static-dist epoch: " + str(epoch + 1), "loss: ", rslt[0])

    #     return rslt[1:]

    # solve in static mode with auto dist
    def __solve_static_auto_dist(self, num_epoch, bs, checkpoint_freq):

        # inputs and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        # number of inputs and labels
        ninputs = len(inputs)
        nlabels = len(labels)

        inputs_labels = inputs + labels  # tmp to one list

        # strategy
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

        model = ModelStatic(self.pde, self.algo, ninputs, inputs_attr, nlabels,
                            labels_attr)

        inputs_labels_spec = list()
        for i in inputs_labels:
            inputs_labels_spec.append(
                InputSpec(i.shape, self._dtype, 'input' + str(i)))

        labels_spec = None

        # engine
        engine = Engine(
            model,
            inputs_spec=inputs_labels_spec,
            labels_spec=labels_spec,
            strategy=dist_strategy)

        # dataset
        train_dataset = DataSetStatic(num_epoch, inputs_labels)

        print("\n ********** engine training start ****  \n")

        # train
        engine.prepare(optimizer=self.opt, loss=loss_func)
        rslt = engine.fit(train_dataset, sample_generator=False)

        print("\n ********** engine predict start ****  \n")

        # predict
        test_dataset = DataSetStatic(1, inputs_labels)
        engine.prepare(optimizer=self.opt, loss=loss_func, mode='predict')
        rslt = engine.predict(test_dataset, sample_generator=False)

        print("\n ********** engine done ****  \n")

        # print(rslt[0][1:])

        return rslt[0][1:]

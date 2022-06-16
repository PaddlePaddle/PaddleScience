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

import jax
import jax.numpy as jnp

paddle.disable_static()

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
        for input in inputs_labels:
            input.stop_gradient = False

        loss, outs = self.algo.compute(
            *inputs_labels,
            ninputs=self.ninputs,
            inputs_attr=self.inputs_attr,
            nlabels=self.nlabels,
            labels_attr=self.labels_attr,
            pde=self.pde)

        return loss, outs  # TODO: add outs


def loss_func(x, y):

    # print("\n ********** loss_func done ****  \n")
    return x


class Solver(object):
    """
    Solver
 
    Parameters:
        algo(Algorithm): The algorithm used in the solver.
        opt(paddlescience.Optimizer): The optimizer used in the solver.

    Example:
        >>> import paddlescience as psci
        >>> solver = psci.solver.Solver(pde=pde_disc, algo=algo, opt=opt)
    """

    # init
    def __init__(self, pde, algo, opt):
        super(Solver, self).__init__()

        self.pde = pde
        self.algo = algo
        self.opt = opt
        self._dtype = config._dtype

        if config._compute_backend == "jax":
            self.__init_jax()
        else:
            if paddle.in_dynamic_mode():
                self.__init_dynamic()
            else:
                if paddle.distributed.get_world_size() == 1:
                    self.__init_static()
                else:
                    self.__init_static_auto_dist()

    # solve (train)
    def solve(self, num_epoch=2, bs=None, checkpoint_freq=1000):
        if config._compute_backend == "jax":
            return self.__solve_jax(num_epoch, bs, checkpoint_freq)
        else:
            if paddle.in_dynamic_mode():
                return self.__solve_dynamic(num_epoch, bs, checkpoint_freq)
            else:
                if paddle.distributed.get_world_size() == 1:
                    return self.__solve_static(num_epoch, bs, checkpoint_freq)
                else:
                    return self.__solve_static_auto_dist(num_epoch, bs,
                                                         checkpoint_freq)

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
        #     num_epoch(int): Optional, default 1000. Number of epochs.
        #     batch_size(int|None): Under develop. Optional, default None. How many sample points are used as a batch during training.
        #     checkpoint_freq(int): Under develop. Optional, default 1000. How many epochs to store the training status once.

        # Return:
        #     solution(Callable): A python func functhion that takes a GeometryDiscrete as input and a numpy array as outputs.

        # Example:
        #     >>> import paddlescience as psci
        #     >>> solver = psci.solver.Solver(algo=algo, opt=opt)
        #     >>> solution = solver.solve(num_epoch=10000)
        #     >>> rslt = solution(geo)
        # """

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        self.inputs = inputs
        self.inputs_attr = inputs_attr
        self.labels = labels
        self.labels_attr = labels_attr

    # solve static 
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

    # predict dynamic
    def __predict_dynamic(self):
        # create inputs 
        inputs, inputs_attr = self.algo.create_inputs(self.pde)

        # convert inputs to tensor
        for i in range(len(inputs)):
            inputs[i] = paddle.to_tensor(
                inputs[i], dtype=self._dtype, stop_gradient=False)

        outs = self.algo.compute_forward(*inputs)

        for i in range(len(outs)):
            outs[i] = outs[i].numpy()

        return outs

    # init static
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

        self.train_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.predict_program = paddle.static.Program()

        # construct train program
        with paddle.static.program_guard(self.train_program,
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

        # construct predict program
        with paddle.static.program_guard(self.predict_program):
            with paddle.utils.unique_name.guard():

                self.algo.net.make_network_static()
                ins = list()
                for i in range(len(inputs)):
                    ishape = list(inputs[i].shape)
                    ishape[0] = -1
                    input = paddle.static.data(
                        name='input' + str(i), shape=ishape, dtype=self._dtype)
                    input.stop_gradient = False
                    ins.append(input)

                self.outs_predict = self.algo.compute_forward(*ins)

        # startup program
        self.exe.run(self.startup_program)

    # solve static
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
            rslt = self.exe.run(self.train_program,
                                feed=feeds,
                                fetch_list=fetches)
            print("static epoch: " + str(epoch + 1), "loss: ", rslt[0])

        return rslt[1:]

    # predict static
    def __predict_static(self):

        # create inputs and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds['input' + str(i)] = inputs[i]

        # fetch outputs
        fetches = list()
        for out in self.outs_predict:
            fetches.append(out.name)

        # run
        rslt = self.exe.run(self.predict_program,
                            feed=feeds,
                            fetch_list=fetches)

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

        inputs_labels = inputs + labels  # tmp to one list

        # strategy
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

        model = ModelStatic(self.pde, self.algo, ninputs, inputs_attr, nlabels,
                            labels_attr)

        inputs_labels_spec = list()
        for i, data in enumerate(inputs_labels):
            inputs_labels_spec.append(
                InputSpec(data.shape, 'float32', 'input' + str(i)))

        labels_spec = None

        # engine
        self.engine = Engine(
            model,
            inputs_spec=inputs_labels_spec,
            labels_spec=labels_spec,
            strategy=dist_strategy)

        # prepare
        self.engine.prepare(
            optimizer=self.opt, loss=loss_func, gradient_scale=False)

    # solve in static mode with auto dist
    def __solve_static_auto_dist(self, num_epoch, bs, checkpoint_freq):

        # inputs and its attributes
        inputs = self.inputs
        inputs_attr = self.inputs_attr
        labels = self.labels
        labels_attr = self.labels_attr

        # dataset
        train_dataset = DataSetStatic(num_epoch, inputs + labels)
        # train
        self.engine.fit(train_dataset, batch_size=None)

        # predict
        self.predict_auto_dist_program = paddle.fluid.Program()
        with paddle.static.program_guard(self.predict_auto_dist_program):
            with paddle.utils.unique_name.guard():

                self.algo.net.make_network_static()
                ins = list()
                for i in range(len(inputs)):
                    ishape = list(inputs[i].shape)
                    ishape[0] = -1
                    input = paddle.static.data(
                        name='input' + str(i), shape=ishape, dtype=self._dtype)
                    input.stop_gradient = False
                    ins.append(input)

                self.outs_predict = self.algo.compute_forward(*ins)

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds['input' + str(i)] = inputs[i]

        # fetch_list
        fetches = []
        for out in self.outs_predict:
            fetches.append(out.name)

        rslt = self.engine._executor.run(self.predict_auto_dist_program,
                                         feed=feeds,
                                         fetch_list=fetches)

        return rslt

    # predict static auto-dist
    def __predict_static_auto_dist(self):

        # create inputs and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)

        # feeds inputs
        feeds = dict()
        for i in range(len(inputs)):
            feeds['input' + str(i)] = inputs[i]

        # fetch_list
        fetches = []
        for out in self.outs_predict:
            fetches.append(out.name)

        rslt = self.engine._executor.run(self.predict_auto_dist_program,
                                         feed=feeds,
                                         fetch_list=fetches)
        return rslt

    def __init_jax(self):

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        self.inputs = inputs
        self.inputs_attr = inputs_attr
        self.labels = labels
        self.labels_attr = labels_attr

        [optim_init, self.optim_update, self.optim_params] = self.opt
        self.optim_state = optim_init(self.algo.net.weights)

    def __solve_jax(self, num_epoch, bs, checkpoint_freq):

        inputs = self.inputs
        inputs_attr = self.inputs_attr
        labels = self.labels
        labels_attr = self.labels_attr

        # number of inputs and labels
        ninputs = len(inputs)
        nlabels = len(labels)

        # convert inputs to tensor
        for i in range(ninputs):
            inputs[i] = jnp.array(inputs[i], dtype=self._dtype)

        # convert label to tensor
        for i in range(nlabels):
            labels[i] = jnp.array(labels[i], dtype=self._dtype)

        inputs_labels = inputs + labels  # tmp to one list

        for epoch in range(num_epoch):

            param = self.optim_params(self.optim_state)

            grads = jax.grad(
                self.algo.compute,
                argnums=6)(*inputs_labels, ninputs, inputs_attr, nlabels,
                           labels_attr, self.pde, param)

            self.optim_state = self.optim_update(epoch, grads,
                                                 self.optim_state)

        return outs

    def feed_data_interior_cur(self, data):
        self.labels = self.algo.feed_data_interior_cur(self.labels,
                                                       self.labels_attr, data)

    def feed_data_user_cur(self, data):
        self.labels = self.algo.feed_data_user_cur(self.labels,
                                                   self.labels_attr, data)

    def feed_data_user_next(self, data):
        self.labels = self.algo.feed_data_user_next(self.labels,
                                                    self.labels_attr, data)

    def feed_data_user(self, data):
        self.feed_data_user_next(data)

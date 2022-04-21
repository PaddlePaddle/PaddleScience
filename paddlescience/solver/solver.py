# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.engine import Engine

__all__ = ["Solver"]


class DataSetStatic:
    def __init__(self, nsamples, inputs_labels):
        self.inputs = inputs
        self.nsamples = nsamples

    def __getitem__(self, idx):
        return self.inputs_labels

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

        _global_process_mesh = auto.ProcessMesh([0, 1])

        for v in inputs_labels:
            auto.shard_tensor(
                v,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [0, -1]
                })

        loss, outs = self.algo.compute(
            *inputs_labels,
            ninputs=self.ninputs,
            inputs_attr=self.inputs_attr,
            nlabels=self.nlabels,
            labels_attr=self.labels_attr,
            pde=self.pde)

        # print("\n ********** compute done ****  \n")

        return loss  #, outs # TODO: add outs


def loss_func(x):

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

    def solve(self, num_epoch=2, bs=None, checkpoint_freq=1000):

        # return self.__solve_static_auto_dist(num_epoch, bs, checkpoint_freq)

        if paddle.in_dynamic_mode():
            return self.__solve_dynamic(num_epoch, bs, checkpoint_freq)
        else:
            return self.__solve_static(num_epoch, bs, checkpoint_freq)

    # solve in dynamic mode
    def __solve_dynamic(self, num_epoch, bs, checkpoint_freq):
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

        # number of inputs and labels
        ninputs = len(inputs)
        nlabels = len(labels)

        # convert inputs to tensor
        for i in range(ninputs):
            inputs[i] = paddle.to_tensor(
                inputs[i], dtype='float32', stop_gradient=False)

        # convert label to tensor
        for i in range(nlabels):
            labels[i] = paddle.to_tensor(
                labels[i], dtype='float32', stop_gradient=False)

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

            # print("epoch/num_epoch: ", epoch + 1, "/", num_epoch,
            #       "batch/num_batch: ", batch_id + 1, "/", num_batch,
            #       "loss: ",
            #       loss.numpy()[0], "eq_loss: ", losses[0].numpy()[0],
            #       "bc_loss: ", losses[1].numpy()[0], "ic_loss: ",
            #       losses[2].numpy()[0])

            # if (epoch + 1) % checkpoint_freq == 0:
            #     paddle.save(self.algo.net.state_dict(),
            #                 './checkpoint/net_params_' + str(epoch + 1))
            #     paddle.save(self.opt.state_dict(),
            #                 './checkpoint/opt_params_' + str(epoch + 1))
            #     if self.algo.loss.geo.time_dependent == False:
            #         np.save(
            #             './checkpoint/rslt_' + str(epoch + 1) + '.npy',
            #             self.algo.net.nn_func(self.algo.loss.geo.space_domain))
            #     else:
            #         np.save('./checkpoint/rslt_' + str(epoch + 1) + '.npy',
            #                 self.algo.net.nn_func(self.algo.loss.geo.domain))

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

    def __solve_static(self, num_epoch, bs, checkpoint_freq):

        # create inputs/labels and its attributes
        inputs, inputs_attr = self.algo.create_inputs(self.pde)
        labels, labels_attr = self.algo.create_labels(self.pde)

        u_n = np.zeros(inputs[0].shape)
        labels = self.algo.feed_labels_u_n(labels, labels_attr, u_n)

        # number of inputs and labels
        ninputs = len(inputs)
        nlabels = len(labels)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        inputs_labels = list()
        feeds = dict()

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        # construct program
        with paddle.static.program_guard(main_program, startup_program):

            # dynamic mode: make network in net's constructor
            # static  mode: make network here 
            self.algo.net.make_network_static()

            # inputs
            for i in range(len(inputs)):
                #inputs
                input = paddle.static.data(
                    name='input' + str(i),
                    shape=inputs[i].shape,
                    dtype='float32')
                input.stop_gradient = False
                inputs_labels.append(input)

            for i in range(len(labels)):
                #labels
                input = paddle.static.data(
                    name='label' + str(i),
                    shape=labels[i].shape,
                    dtype='float32')
                label.stop_gradient = False
                inputs_labels.append(label)

            loss, outs = self.algo.compute(
                *inputs_labels,
                ninputs=ninputs,
                inputs_attr=inputs_attr,
                nlabels=nlabels,
                labels_attr=labels_attr,
                pde=self.pde)

            self.opt.minimize(loss)

        # feeds inputs
        for i in range(len(inputs)):
            feeds['input' + str(i)] = inputs[i]

        # feeds labels
        for i in range(len(labels)):
            feeds['label' + str(i)] = labels[i]

        # fetch loss and net's outputs
        fetches = [loss.name]
        for out in outs:
            fetches.append(out.name)

        # start up program
        exe.run(startup_program)

        # main loop
        for epoch in range(num_epoch):
            rslt = exe.run(main_program, feed=feeds, fetch_list=fetches)
            print("static epoch: " + str(epoch + 1), "loss: ", rslt[0])

        return rslt[1:]

    def __solve_static_dist(self, num_epoch, bs, checkpoint_freq):

        # init dist environment
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

        inputs, inputs_attr = self.algo.create_inputs(self.pde)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        # dist optimizer
        opt_dist = fleet.distributed_optimizer(self.opt)

        inputs = list()
        feeds = dict()

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        # construct program
        with paddle.static.program_guard(main_program, startup_program):

            self.algo.net.make_network_static()

            for i in range(len(inputs)):
                # inputs
                input = paddle.static.data(
                    name='input' + str(i),
                    shape=inputs[i].shape,
                    dtype='float32')
                input.stop_gradient = False
                inputs.append(input)

                # feeds
                feeds['input' + str(i)] = inputs[i]

            loss, outs = self.algo.compute(
                *inputs, inputs_attr=inputs_attr, pde=self.pde)

            opt_dist.minimize(loss)

        # fetch loss and net's output
        fetches = [loss.name]
        for out in outs:
            fetches.append(out.name)

        # start up program
        exe.run(startup_program)

        # main loop
        for epoch in range(num_epoch):
            rslt = exe.run(main_program, feed=feeds, fetch_list=fetches)
            print("static-dist epoch: " + str(epoch + 1), "loss: ", rslt[0])

        return rslt[1:]

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
                InputSpec(i.shape, 'float32', 'input' + str(i)))

        labels_spec = None

        engine = Engine(
            model,
            inputs_spec=inputs_labels_spec,
            labels_spec=labels_spec,
            strategy=dist_strategy)

        print("\n ********** engine prepare start ****  \n")

        engine.prepare(optimizer=self.opt, loss=loss_func)

        print("\n ********** engine prepare done ****  \n")

        train_dataset = DataSetStatic(num_epoch, inputs_labels)
        rslt = engine.fit(train_dataset, sample_generator=False)

        print("\n ********** engine rslt done ****  \n")

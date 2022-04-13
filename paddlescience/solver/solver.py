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
    def __init__(self, nsamples, ins):
        self.ins = ins
        self.nsamples = nsamples

    def __getitem__(self, idx):
        return self.ins

    def __len__(self):
        return self.nsamples


class ModelStatic(paddle.nn.Layer):
    def __init__(self, pde, algo, ins_attr):
        super(ModelStatic, self).__init__()
        self.pde = pde
        self.algo = algo
        self.ins_attr = ins_attr
        self.algo.net.make_network_static()

    def forward(self, *args):

        ins_attr = self.ins_attr

        _global_process_mesh = auto.ProcessMesh([0])

        n = 0
        for attr in ins_attr["interior"].values():
            input = args[n]
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, -1]
                })
            n += 1

        for attr in ins_attr["boundary"].values():
            input = args[n]
            auto.shard_tensor(
                input,
                dist_attr={
                    "process_mesh": _global_process_mesh,
                    "dims_mapping": [-1, -1]
                })
            n += 1

        loss, outs = self.algo.compute(*args, ins_attr=ins_attr, pde=self.pde)

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

    # solver in static mode
    def solve_static(self, num_epoch=1, bs=None, checkpoint_freq=1000):

        ins, ins_attr = self.algo.create_ins(self.pde)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        inputs = list()
        feeds = dict()

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        # 
        with paddle.static.program_guard(main_program, startup_program):

            self.algo.net.make_network_static()

            for i in range(len(ins)):
                #inputs
                input = paddle.static.data(
                    name='input' + str(i), shape=ins[i].shape, dtype='float32')
                input.stop_gradient = False
                inputs.append(input)

                # feeds
                feeds['input' + str(i)] = ins[i]

                # print(ins[i])

            loss, outs = self.algo.compute(
                *inputs, ins_attr=ins_attr, pde=self.pde)

            self.opt.minimize(loss)

        # fetch loss and net's output
        fetches = [loss.name]
        for out in outs:
            fetches.append(out.name)

        # start up program
        exe.run(startup_program)

        # print(feeds)

        # main loop
        for i in range(num_epoch):
            rslt = exe.run(main_program, feed=feeds, fetch_list=fetches)
            print("loss: ", rslt[0])

    def solve_static_dist(self, num_epoch=2, bs=None, checkpoint_freq=1000):

        # init dist environment
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

        ins, ins_attr = self.algo.create_ins(self.pde)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        # dist optimizer
        opt_dist = fleet.distributed_optimizer(self.opt)

        inputs = list()
        feeds = dict()

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        # 
        with paddle.static.program_guard(main_program, startup_program):

            self.algo.net.make_network_static()

            for i in range(len(ins)):
                #inputs
                input = paddle.static.data(
                    name='input' + str(i), shape=ins[i].shape, dtype='float32')
                input.stop_gradient = False
                inputs.append(input)

                # feeds
                feeds['input' + str(i)] = ins[i]

            loss, outs = self.algo.compute(
                *inputs, ins_attr=ins_attr, pde=self.pde)

            opt_dist.minimize(loss)

        # fetch loss and net's output
        fetches = [loss.name]
        for out in outs:
            fetches.append(out.name)

        # start up program
        exe.run(startup_program)

        # main loop
        for i in range(num_epoch):
            rslt = exe.run(main_program, feed=feeds, fetch_list=fetches)
            print("loss: ", rslt[0])

    # solve in static mode with auto dist
    def solve_static_auto_dist(self,
                               num_epoch=2,
                               bs=None,
                               checkpoint_freq=1000):

        # inputs and its attributes
        ins, ins_attr = self.algo.create_ins(self.pde)

        # strategy
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)

        model = ModelStatic(self.pde, self.algo, ins_attr)

        inputs_spec = list()
        for i in ins:
            inputs_spec.append(InputSpec(i.shape, 'float32', 'input' + str(i)))

        labels_spec = None

        engine = Engine(
            model,
            inputs_spec=inputs_spec,
            labels_spec=labels_spec,
            strategy=dist_strategy)

        print("\n ********** engine prepare start ****  \n")

        engine.prepare(optimizer=self.opt, loss=loss_func)

        print("\n ********** engine prepare done ****  \n")

        train_dataset = DataSetStatic(num_epoch, ins)
        rslt = engine.fit(train_dataset)

        print("\n ********** engine rslt done ****  \n")

    # solve in dynamic mode
    def solve_dynamic(self, num_epoch=2, bs=None, checkpoint_freq=1000):
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

        ins, ins_attr = self.algo.create_ins(self.pde)

        # print(ins)

        # convert ins to tensor
        for i in range(len(ins)):
            ins[i] = paddle.to_tensor(
                ins[i], dtype='float32', stop_gradient=False)

        # make network
        # self.algo.net.make_network()

        for epoch in range(num_epoch):

            loss, outs = self.algo.compute(
                *ins, ins_attr=ins_attr, pde=self.pde)

            loss.backward()
            self.opt.step()
            self.opt.clear_grad()

            print("epoch: " + str(epoch + 1), "    loss:", loss.numpy()[0])

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

        def solution_fn(geo):
            if geo.time_dependent == False:
                if not isinstance(geo.space_domain, paddle.Tensor):
                    geo.set_batch_size(geo.get_domain_size())
                    geo.to_tensor()
                return self.algo.net.nn_func(geo.space_domain).numpy()
            else:
                if not isinstance(geo.domain, paddle.Tensor):
                    geo.set_batch_size(geo.get_domain_size())
                    geo.to_tensor()
                return self.algo.net.nn_func(geo.domain).numpy()

        return solution_fn

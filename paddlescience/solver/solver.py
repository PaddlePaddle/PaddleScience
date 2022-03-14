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

__all__ = ["Solver"]


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

    def __init__(self, algo, opt):
        super(Solver, self).__init__()

        self.algo = algo
        self.opt = opt

    def solve(self, num_epoch=1000, batch_size=None, checkpoint_freq=1000):
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
        batch_size = self.algo.loss.geo.get_domain_size(
        ) if batch_size is None else batch_size
        self.algo.loss.set_batch_size(batch_size)
        self.algo.loss.pdes.to_tensor()
        self.algo.loss.geo.to_tensor()
        num_batch = self.algo.loss.num_batch

        for epoch_id in range(num_epoch):
            for batch_id in range(num_batch):
                loss, losses = self.algo.batch_run(batch_id)
                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
                print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
                      "batch/num_batch: ", batch_id + 1, "/", num_batch,
                      "loss: ",
                      loss.numpy()[0], "eq_loss: ", losses[0].numpy()[0],
                      "bc_loss: ", losses[1].numpy()[0], "ic_loss: ",
                      losses[2].numpy()[0])
            if (epoch_id + 1) % checkpoint_freq == 0:
                paddle.save(self.algo.net.state_dict(),
                            './checkpoint/net_params_' + str(epoch_id + 1))
                paddle.save(self.opt.state_dict(),
                            './checkpoint/opt_params_' + str(epoch_id + 1))
                if self.algo.loss.geo.time_dependent == False:
                    np.save(
                        './checkpoint/rslt_' + str(epoch_id + 1) + '.npy',
                        self.algo.net.nn_func(self.algo.loss.geo.space_domain))
                else:
                    np.save('./checkpoint/rslt_' + str(epoch_id + 1) + '.npy',
                            self.algo.net.nn_func(self.algo.loss.geo.domain))

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

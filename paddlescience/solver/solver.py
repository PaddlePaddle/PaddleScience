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


class Solver(object):
    def __init__(self, algo, opt):
        super(Solver, self).__init__()

        self.algo = algo
        self.opt = opt

    def solve(self, num_epoch=1, batch_size=None, checkpoint_freq=1000):
        batch_size = self.algo.loss.geo.get_nsteps(
        ) if batch_size is None else batch_size
        self.algo.loss.set_batch_size(batch_size)
        self.algo.loss.pdes.to_tensor()
        self.algo.loss.geo.to_tensor()
        num_batch = self.algo.loss.num_batch

        for epoch_id in range(num_epoch):
            for batch_id in range(num_batch):
                eq_loss, bc_loss, ic_loss = self.algo.batch_run(batch_id)
                loss = eq_loss + bc_loss + ic_loss
                loss.backward()
                self.opt.step()
                self.opt.clear_grad()
                print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
                      "batch/num_batch: ", batch_id + 1, "/", num_batch,
                      "loss: ", loss.numpy()[0], "eq_loss: ",
                      eq_loss.numpy()[0], "bc_loss: ", bc_loss.numpy()[0])
            if epoch_id % checkpoint_freq == 0:
                np.save('./checkpoint_' + str(epoch_id) + '.npy',
                        self.algo.net.nn_func(self.algo.loss.geo.steps))

        def solution_fn(geo):
            return self.algo.net.nn_func(geo.steps)

        return solution_fn

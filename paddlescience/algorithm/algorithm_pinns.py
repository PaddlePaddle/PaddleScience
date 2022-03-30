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

import paddle
from .algorithm_base import AlgorithmBase
from ..ins import InsDataWithAttr


class PINNs(AlgorithmBase):
    """
    The Physics Informed Neural Networks Algorithm.

    Parameters:
        net(NetworkBase): The NN network used in PINNs algorithm.
        loss(LossBase): The loss used in PINNs algorithm.

    Example:
        >>> import paddlescience as psci
        >>> algo = psci.algorithm.PINNs(net=net, loss=loss)
    """

    def __init__(self, net, loss):
        super(PINNs, self).__init__()
        self.net = net
        self.loss = loss

    def create_ins(self, pde):
        ins = dict()

        # TODO: hard code
        ins_i = dict()
        points = pde.geometry.interior
        data = points  # paddle.to_tensor(points, dtype='float32', stop_gradient=False)
        ins_i["0"] = InsDataWithAttr(data, 0, 0)
        ins["interior"] = ins_i

        ins_b = dict()
        for name, points in pde.geometry.boundary.items():
            data = points  #paddle.to_tensor(
            # points, dtype='float32', stop_gradient=False)
            ins_b[name] = InsDataWithAttr(data, 0, 0)
        ins["boundary"] = ins_b

        return ins

    def compute(self, ins, pde):

        # interior out and loss
        for input in ins["interior"].values():
            loss_i, outs = self.loss.eq_loss(
                pde, self.net, input, bs=4)  # TODO bs
            loss = loss_i  # TODO: += 1

        # boundary out and loss
        for input in ins["boundary"].values():
            loss_b, outs = self.loss.bc_loss(
                pde, self.net, input, bs=2)  # TODO bs
            loss += loss_b

        return loss  # TODO: return more

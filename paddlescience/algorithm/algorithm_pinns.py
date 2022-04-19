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
from ..ins import InsAttr


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

        ins = list()
        ins_attr = dict()

        # TODO: hard code
        ins_attr_i = dict()
        points = pde.geometry.interior
        data = points  # 
        ins.append(data)
        ins_attr_i["0"] = InsAttr(0, 0)
        ins_attr["interior"] = ins_attr_i

        ins_attr_b = dict()
        # loop on bc
        for name in pde.bc.keys():
            data = pde.geometry.boundary[name]
            ins.append(data)
            ins_attr_b[name] = InsAttr(0, 0)
        ins_attr["boundary"] = ins_attr_b

        return ins, ins_attr

    def compute(self, *ins, ins_attr, pde):

        # print(args[1])

        outs = list()

        # interior outputs and loss
        n = 0
        loss = 0.0
        for name_i, input_attr in ins_attr["interior"].items():
            input = ins[n]
            loss_i, out_i = self.loss.eq_loss(
                pde, self.net, name_i, input, input_attr,
                bs=-1)  # TODO: bs is not used
            loss += loss_i
            outs.append(out_i)
            n += 1

        # boundary outputs and loss
        for name_b, input_attr in ins_attr["boundary"].items():
            input = ins[n]
            loss_b, out_b = self.loss.bc_loss(
                pde, self.net, name_b, input, input_attr,
                bs=-1)  # TODO: bs is not used
            loss += loss_b
            outs.append(out_b)
            n += 1

        loss = paddle.sqrt(loss)
        return loss, outs  # TODO: return more

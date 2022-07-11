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

import paddle
import copy
from .loss_base import LossFormula, CompFormula


class IcLoss(LossFormula):
    def __init__(self):
        super(IcLoss, self).__init__()
        self._loss_obj = [self]

    def compute(self, pde, net, input, rhs=None):

        # compute outs
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs(input, bs)
        # loss
        loss = 0.0
        for i in range(len(pde.ic)):
            formula = pde.ic[i].formula
            rst = cmploss.compute_formula(formula, input)
            # loss 
            if rhs is None:
                loss += paddle.norm(rst**2, p=1) * self._loss_wgt
            else:
                loss += paddle.norm((rst - rhs)**2, p=1) * self._loss_wgt

        return loss, cmploss.outs

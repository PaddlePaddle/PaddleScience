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


class BcLoss(LossFormula):
    def __init__(self, name):
        super(BcLoss, self).__init__()
        self._loss_obj = [self]
        self._name = name

    def compute(self, pde, net, input, rhs=None):

        # compute outs, jacobian, hessian
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs_der(input, bs)

        loss = 0.0
        for i in range(pde.bc[self._name]):
            # compute rst on left-hand side
            formula = pde.bc[self._name][i].formula
            rst = cmploss.compute_formula(formula, input)
            # loss 
            if rhs is None:
                loss += paddle.norm(rst**2, p=1) * self._loss_wgt
            else:
                loss += paddle.norm((rst - rhs)**2, p=1) * self._loss_wgt

        return loss, cmploss.outs

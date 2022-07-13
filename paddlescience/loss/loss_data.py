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
from .loss_base import FormulaLoss, CompFormula


class DataLoss(FormulaLoss):
    def __init__(self):
        super(DataLoss, self).__init__()
        self._loss = [self]

    def compute(self, pde, net, input):

        # compute outs
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs(input, bs)
        # loss
        loss = 0.0
        for i in range(len(pde.dvar)):
            idx = labels_attr["data_next"][i]
            data = labels[idx]
            loss += paddle.norm(cmploss.outs[:, i] - data, p=2)**2

        loss = self.data_weight * loss
        return loss, cmploss.outs

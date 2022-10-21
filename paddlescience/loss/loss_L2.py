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
import numpy as np
from .loss_base import CompFormula, l2_norm_square
from ..labels import LabelInt
from .. import config


class L2:
    """
    L2 loss.

    Parameters:
        p(1 or 2):

            p=1: total loss = eqloss + bcloss + icloss + dataloss.

            p=2: total loss = sqrt(eqloss**2 + bcloss**2 + icloss**2 + dataloss**2)

    Example:
        >>> import paddlescience as psci
        >>> loss = psci.loss.L2()
    """

    def __init__(self,
                 p=1,
                 eq_weight=None,
                 bc_weight=None,
                 ic_weight=None,
                 data_weight=1.0):
        self.norm_p = p
        self.eq_weight = eq_weight
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight
        self.data_weight = data_weight

        #TODO: check input

    # compute loss on one interior 
    # there are multiple pde

    def eq_loss(self,
                pde,
                net,
                input,
                input_attr,
                labels,
                labels_attr,
                bs,
                params=None):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(input, bs, params)

        # print(input)
        # print(cmploss.outs[0:4,:])

        loss = 0.0
        for i in range(len(pde.equations)):
            formula = pde.equations[i]
            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, None, params)

            # TODO: simplify
            rhs_eq = labels_attr["equations"][i]["rhs"]
            if type(rhs_eq) == LabelInt:
                rhs = labels[rhs_eq]
            else:
                rhs = rhs_eq

            if self.eq_weight is None:
                wgt_eq = labels_attr["equations"][i]["weight"]
            else:
                if np.isscalar(self.eq_weight):
                    wgt_eq = self.eq_weight
                else:
                    wgt_eq = self.eq_weight[i]

            if wgt_eq is None:
                wgt = None
            elif type(wgt_eq) == LabelInt:
                wgt = labels[wgt_eq]
            elif np.isscalar(wgt_eq):
                wgt = wgt_eq
            else:
                pass
                # TODO: error out

            if rhs is None:
                loss += l2_norm_square(rst, wgt)
            else:
                loss += l2_norm_square((rst - rhs), wgt)

        return loss, cmploss.outs

    # compute loss on one boundary
    # there are multiple bc on one boundary

    def bc_loss(self,
                pde,
                net,
                name_b,
                input,
                input_attr,
                labels,
                labels_attr,
                bs,
                params=None):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(input, bs,
                                 params)  # TODO: dirichlet not need der

        loss = 0.0
        for i in range(len(pde.bc[name_b])):
            # TODO: hard code bs

            normal_b = labels_attr["bc"][name_b][i]["normal"]
            if type(normal_b) == LabelInt:
                normal = labels[normal_b]
            else:
                normal = normal_b

            formula = pde.bc[name_b][i].formula
            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, normal, params)

            # TODO: simplify                                  
            rhs_b = labels_attr["bc"][name_b][i]["rhs"]
            if type(rhs_b) == LabelInt:
                rhs = labels[rhs_b]
            else:
                rhs = rhs_b

            if self.bc_weight is None:
                wgt_b = labels_attr["bc"][name_b][i]["weight"]
            else:
                if np.isscalar(self.bc_weight):
                    wgt_b = self.bc_weight
                else:
                    wgt_b = self.bc_weight[i]

            if wgt_b is None:
                wgt = None
            elif type(wgt_b) == LabelInt:
                wgt = labels[wgt_b]
            else:
                wgt = wgt_b

            if rhs is None:
                loss += l2_norm_square(rst, wgt)
            else:
                loss += l2_norm_square((rst - rhs), wgt)

        return loss, cmploss.outs

    def ic_loss(self,
                pde,
                net,
                input,
                input_attr,
                labels,
                labels_attr,
                bs,
                params=None):

        # compute outs
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs(input, bs, params)

        loss = 0.0
        for i in range(len(pde.ic)):
            formula = pde.ic[i].formula
            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, None, params)

            rhs_c = labels_attr["ic"][i]["rhs"]
            if type(rhs_c) == LabelInt:
                rhs = labels[rhs_c]
            else:
                rhs = rhs_c

            if self.ic_weight is None:
                wgt = labels_attr["ic"][i]["weight"]
            else:
                wgt = self.ic_weight
            loss += l2_norm_square(rst - rhs, wgt)

        return loss, cmploss.outs

    # compute loss on real data 
    def data_loss(self,
                  pde,
                  net,
                  input,
                  input_attr,
                  labels,
                  labels_attr,
                  bs,
                  params=None):

        cmploss = CompFormula(pde, net)

        # compute outs
        cmploss.compute_outs(input, bs, params)

        loss = 0.0
        for i in range(len(pde.dvar)):
            idx = labels_attr["data_next"][i]
            data = labels[idx]
            if config.prim_enabled():
                nrm = paddle.norm(cmploss.outs[:, i] - data, p=2)
                loss += nrm * nrm
            else:
                loss += paddle.norm(cmploss.outs[:, i] - data, p=2)**2

            # TODO: p=2 p=1

        loss = self.data_weight * loss
        return loss, cmploss.outs

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


class FormulaLoss:
    def __init__(self):
        self._eqlist = list()
        self._bclist = list()
        self._iclist = list()
        self._datalist = list()

        self._loss_wgt = 1.0

        self.norm_p = 1

    # add class
    def __add__(self, other):
        floss = FormulaLoss()
        floss._eqlist = self._eqlist + other._eqlist
        floss._bclist = self._bclist + other._bclist
        floss._iclist = self._iclist + other._iclist
        floss._datalist = self._datalist + other._datalist
        return floss

    # multiply scalar (right)
    def __mul__(self, other):
        floss = copy.deepcopy(self)
        floss._loss_wgt *= self._loss_wgt * other
        return floss

    # multiply scalar (left)
    def __rmul__(self, other):
        floss = copy.deepcopy(self)
        floss._loss_wgt *= self._loss_wgt * other
        return floss

    def eq_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(input, bs)

        # print(input)
        # print(cmploss.outs[0:4,:])

        loss = 0.0
        for i in range(len(pde.equations)):
            formula = pde.equations[i]
            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, None)

            # TODO: simplify
            rhs_eq = labels_attr["equations"][i]["rhs"]
            if type(rhs_eq) == LabelInt:
                rhs = labels[rhs_eq]
            else:
                rhs = rhs_eq

            wgt_eq = labels_attr["equations"][i]["weight"]
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
    def bc_loss(self, pde, net, name_b, input, input_attr, labels, labels_attr,
                bs):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(input, bs)  # TODO: dirichlet not need der

        loss = 0.0
        for i in range(len(pde.bc[name_b])):
            # TODO: hard code bs
            formula = pde.bc[name_b][i].formula
            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, None)

            # TODO: simplify                                  
            rhs_b = labels_attr["bc"][name_b][i]["rhs"]
            if type(rhs_b) == LabelInt:
                rhs = labels[rhs_b]
            else:
                rhs = rhs_b

            wgt_b = labels_attr["bc"][name_b][i]["weight"]
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

    def ic_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

        # compute outs
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs(input, bs)

        loss = 0.0
        for i in range(len(pde.ic)):
            formula = pde.ic[i].formula
            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, None)

            rhs_c = labels_attr["ic"][i]["rhs"]
            if type(rhs_c) == LabelInt:
                rhs = labels[rhs_c]
            else:
                rhs = rhs_c
            wgt = labels_attr["ic"][i]["weight"]
            loss += l2_norm_square(rst - rhs, wgt)

        return loss, cmploss.outs

    # compute loss on real data 
    def data_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

        cmploss = CompFormula(pde, net)

        # compute outs
        cmploss.compute_outs(input, bs)

        loss = 0.0
        for i in range(len(pde.dvar)):
            idx = labels_attr["data_next"][i]
            data = labels[idx]
            loss += paddle.norm(cmploss.outs[:, i] - data, p=2)**2
            # TODO: p=2 p=1

        loss = self.data_weight * loss
        return loss, cmploss.outs


def EqLoss(eq, netout=None):

    floss = FormulaLoss()
    floss._eqlist = [eq]
    return floss


def BcLoss(name, netout=None):
    floss = FormulaLoss()
    floss._bclist = [name]
    return floss


def IcLoss():
    floss = FormulaLoss()
    return floss


def DataLoss():
    floss = FormulaLoss()
    return floss

    # if netout is not None:
    #     self._net = netout._net
    #     self._input = netout._input

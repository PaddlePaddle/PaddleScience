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
import copy


class FormulaLoss:
    def __init__(self):
        self._eqlist = list()
        self._bclist = list()
        self._iclist = False
        self._datalist = False

        self._eqwgt = list()
        self._bcwgt = list()
        self._icwgt = list()
        self._datawgt = list()

        self._eqinput = list()
        self._eqnet = list()

        self.norm_p = 1

    # add class
    def __add__(self, other):
        floss = FormulaLoss()
        floss._eqlist = self._eqlist + other._eqlist
        floss._bclist = self._bclist + other._bclist
        floss._iclist = self._iclist + other._iclist
        floss._datalist = self._datalist + other._datalist

        floss._eqwgt = self._eqwgt + other._eqwgt
        floss._bcwgt = self._bcwgt + other._bcwgt
        floss._icwgt = self._icwgt + other._icwgt
        floss._datawgt = self._datawgt + other._datawgt

        floss._eqinput = self._eqinput + other._eqinput
        floss._bcinput = self._bcinput + other._bcinput
        floss._icinput = self._icinput + other._icinput
        floss._datainput = self._datainput + other._datainput

        floss._eqnet = self._eqnet + other._eqnet
        floss._bcnet = self._bcnet + other._bcnet
        floss._icnet = self._icnet + other._icnet
        floss._datanet = self._datanet + other._datanet

        return floss

    # multiply scalar (right)
    def __mul__(self, weight):
        floss = copy.deepcopy(self)
        for i in range(len(floss._eqwgt)):
            floss._eqwgt[i] *= weight
        for i in range(len(floss._bcwgt)):
            floss._bcwgt[i] *= weight
        for i in range(len(floss._icwgt)):
            floss._icwgt[i] *= weight
        for i in range(len(floss._datawgt)):
            floss._datawgt[i] *= weight
        return floss

    # multiply scalar (left)
    def __rmul__(self, other):
        floss = copy.deepcopy(self)
        self.__mul__(other)
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

            if formula not in self._eqlist:
                continue
            else:
                idx = self._eqlist.index(formula)

            rst = cmploss.compute_formula(formula, input, input_attr, labels,
                                          labels_attr, None)

            # TODO: simplify
            rhs_eq = labels_attr["equations"][i]["rhs"]
            if type(rhs_eq) == LabelInt:
                rhs = labels[rhs_eq]
            else:
                rhs = rhs_eq

            wgt = self._eqwgt[idx]

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

            idx = self._bclist.index(name_b)
            wgt = self._bcwgt[idx]

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
    floss._eqwgt = [1.0]
    if netout is not None:
        floss._eqinput = [netout._input]
        floss._eqnet = [netout._net]
    return floss


def BcLoss(name, netout=None):
    floss = FormulaLoss()
    floss._bclist = [name]
    floss._bcwgt = [1.0]
    if netout is not None:
        floss._bcinput = [netout._input]
        floss._bcnet = [netout._net]
    return floss


def IcLoss(netout=None):
    floss = FormulaLoss()
    floss._iclist = True
    floss._icwgt = [1.0]
    if netout is not None:
        floss._icinput = [netout._input]
        floss._icnet = [netout._net]
    return floss


def DataLoss(netout=None):
    floss = FormulaLoss()
    floss._datalist = True
    floss._datawgt = [1.0]
    if netout is not None:
        floss._datainput = [netout._input]
        floss._datanet = [netout._net]
    return floss

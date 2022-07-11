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
from .loss_base import LossBase, CompFormula
from ..labels import LabelInt

import copy


# Equation loss
class EqLoss(FormulaLoss):
    def __init__(self, n):
        super(EqLoss, self).__init__()
        self._loss_obj = [self]
        self._n = n

    def compute(self, pde, net, input, rhs=None):

        # compute outs, jacobian, hessian
        cmploss = CompFormula(pde, net)
        cmploss.compute_outs_der(input, bs)

        # compute rst on left-hand side
        formula = pde.equations[self._n]
        rst = cmploss.compute_formula(formula, input)

        # loss
        if rhs is None:
            loss = paddle.norm(rst, p=2) * self._loss_wgt[i]
        else:
            loss = paddle.norm(rst - rhs, p=2) * self._loss_wgt[i]

        return loss, cmploss.outs


class BcLoss(FormulaLoss):
    def __init__(self, name, n=None):
        super(BcLoss, self).__init__()
        self._loss_obj = [self]
        self._name = name
        self._n = n

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


class IcLoss(FormulaLoss):
    def __init__(self):
        super(IcLoss, self).__init__(n)
        self._loss_obj = [self]
        self._n = n

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


class DataLoss(FormulaLoss):
    def __init__(self):
        super(DataLoss, self).__init__()
        self._loss_obj = [self]

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


# eqloss1 = EqLoss()
# eqloss2 = EqLoss()
# bcloss1 = BcLoss("top")
# loss = eqloss1 + eqloss2 + bcloss1  # loss._loss_obj = [eqloss1, eqloss2, bcloss1]

# rst = loss.compute(net, pde, input)
# print(rst)

# # option 1
# eq_loss = psci.loss.Eqloss()
# bc_loss = psci.loss.BcLoss("top")
# total_loss = [3.0, 2.0] * eq_loss + bc_loss

# # option 2
# eq1_loss = psci.loss.Eqloss(1)
# eq2_loss = psci.loss.Eqloss(2)
# bc_loss = psci.loss.BcLoss("top")
# total_loss = 3.0 * eq1_loss + 2.0 * eq2_loss + bc_loss

# print(loss._loss_obj)

# def compute(self, pde, net, input, rhs=None):

#     # compute outs, jacobian, hessian
#     cmploss = CompFormula(pde, net)
#     cmploss.compute_outs_der(input, bs)

#     # compute rst on left-hand side
#     formula = pde.equations[self._neq]
#     rst = cmploss.compute_formula(formula, input)

#     # loss
#     if rhs is None:
#         loss = paddle.norm(rst, p=2)
#     else:
#         loss = paddle.norm(rst - rhs, p=2)

#     return loss, cmploss.outs

# def bc_loss(self, pde, net, input, rhs=None):

#     # compute outs, jacobian, hessian
#     cmploss = CompFormula(pde, net)
#     cmploss.compute_outs_der(input, bs)

#     loss = 0.0
#     for i in range(pde.bc[self._name])):
#         # compute rst on left-hand side
#         formula = pde.bc[self._name][i].formula
#         rst = cmploss.compute_formula(formula, input)

#         # loss 
#         if rhs is None:
#             loss += paddle.norm(rst**2, p=1)
#         else:
#             loss += paddle.norm((rst - rhs)**2, p=1)

#     return paddle.sqrt(loss), cmploss.outs

# class IcLoss():

#     def compute(self, pde, net, input, rhs=None):

# class DataLoss():

#     def compute(self, pde, net, input, rhs=None):

# class L2(LossBase):
#     """
#     L2 loss.

#     Parameters:
#         p(1 or 2):

#             p=1: total loss = eqloss + bcloss + icloss + dataloss.

#             p=2: total loss = sqrt(eqloss**2 + bcloss**2 + icloss**2 + dataloss**2)

#     Example:
#         >>> import paddlescience as psci
#         >>> loss = psci.loss.L2()
#     """

#     def __init__(self, p=1, data_weight=1.0):
#         self.norm_p = p
#         self.data_weight = data_weight

#     # compute loss on one interior 
#     # there are multiple pde
#     def eq_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

#         cmploss = CompFormula(pde, net)

#         # compute outs, jacobian, hessian
#         cmploss.compute_outs_der(input, bs)

#         loss = 0.0
#         for i in range(len(pde.equations)):
#             formula = pde.equations[i]
#             rst = cmploss.compute_formula(formula, input, input_attr, labels,
#                                           labels_attr, None)

#             # TODO: simplify
#             rhs_eq = labels_attr["equations"][i]["rhs"]
#             if type(rhs_eq) == LabelInt:
#                 rhs = labels[rhs_eq]
#             else:
#                 rhs = rhs_eq

#             wgt_eq = labels_attr["equations"][i]["weight"]
#             if wgt_eq is None:
#                 wgt = None
#             elif type(wgt_eq) == LabelInt:
#                 wgt = labels[wgt_eq]
#             elif np.isscalar(wgt_eq):
#                 wgt = wgt_eq
#             else:
#                 pass
#                 # TODO: error out

#             if rhs is None:
#                 if wgt is None:
#                     loss += paddle.norm(rst**2, p=1)
#                 else:
#                     loss += paddle.norm(rst**2 * wgt, p=1)
#             else:
#                 if wgt is None:
#                     loss += paddle.norm((rst - rhs)**2, p=1)
#                 else:
#                     loss += paddle.norm((rst - rhs)**2 * wgt, p=1)

#         return loss, cmploss.outs

#     # compute loss on one boundary
#     # there are multiple bc on one boundary
#     def bc_loss(self, pde, net, name_b, input, input_attr, labels, labels_attr,
#                 bs):

#         cmploss = CompFormula(pde, net)

#         # compute outs, jacobian, hessian
#         cmploss.compute_outs_der(input, bs)  # TODO: dirichlet not need der

#         loss = 0.0
#         for i in range(len(pde.bc[name_b])):
#             # TODO: hard code bs
#             formula = pde.bc[name_b][i].formula
#             rst = cmploss.compute_formula(formula, input, input_attr, labels,
#                                           labels_attr, None)

#             # TODO: simplify                                  
#             rhs_b = labels_attr["bc"][name_b][i]["rhs"]
#             if type(rhs_b) == LabelInt:
#                 rhs = labels[rhs_b]
#             else:
#                 rhs = rhs_b

#             wgt_b = labels_attr["bc"][name_b][i]["weight"]
#             if wgt_b is None:
#                 wgt = None
#             elif type(wgt_b) == LabelInt:
#                 wgt = labels[wgt_b]
#             else:
#                 wgt = wgt_b

#             # print("rst: ", rst.shape)
#             # print("rhs: ", rhs.shape)

#             if rhs is None:
#                 if wgt is None:
#                     loss += paddle.norm(rst**2, p=1)
#                 else:
#                     loss += paddle.norm(rst**2 * wgt, p=1)
#             else:
#                 if wgt is None:
#                     loss += paddle.norm((rst - rhs)**2, p=1)
#                 else:
#                     loss += paddle.norm((rst - rhs)**2 * wgt, p=1)

#         return loss, cmploss.outs

#     def ic_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

#         # compute outs
#         cmploss = CompFormula(pde, net)
#         cmploss.compute_outs(input, bs)

#         loss = 0.0
#         for i in range(len(pde.ic)):
#             formula = pde.ic[i].formula
#             rst = cmploss.compute_formula(formula, input, input_attr, labels,
#                                           labels_attr, None)

#             rhs_c = labels_attr["ic"][i]["rhs"]
#             if type(rhs_c) == LabelInt:
#                 rhs = labels[rhs_c]
#             else:
#                 rhs = rhs_c
#             wgt = labels_attr["ic"][i]["weight"]
#             loss += paddle.norm((rst - rhs)**2 * wgt, p=1)

#         return loss, cmploss.outs

#     # compute loss on real data 
#     def data_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

#         cmploss = CompFormula(pde, net)

#         # compute outs
#         cmploss.compute_outs(input, bs)

#         loss = 0.0
#         for i in range(len(pde.dvar)):
#             idx = labels_attr["data_next"][i]
#             data = labels[idx]
#             loss += paddle.norm(cmploss.outs[:, i] - data, p=2)**2
#             # TODO: p=2 p=1

#         loss = self.data_weight * loss
#         return loss, cmploss.outs

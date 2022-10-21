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
import sympy
import numpy as np
from ..inputs import InputsAttr
from .. import config

from paddle.incubate.autograd import Jacobian, Hessian


class CompFormula:
    def __init__(self, pde, net):
        self.pde = pde
        self.net = net
        self.indvar = pde.indvar
        self.dvar = pde.dvar
        self.dvar_n = pde.dvar_n
        self.outs = None
        self.jacobian = None
        self.hessian = None

    def compute_outs(self, input, bs, params=None):
        self.outs = self.net.nn_func(input, params)

    def compute_outs_der(self, input, bs, params=None):

        # outs
        self.compute_outs(input, bs, params)

        # jacobian
        jacobian = Jacobian(self.net.nn_func, input, is_batched=True)

        # hessian
        hessian = list()
        for i in range(self.net.num_outs):

            def func(input):
                return self.net.nn_func(input)[:, i:i + 1]

            hessian.append(Hessian(func, input, is_batched=True))

        # print("*** Jacobian *** ")
        # print(jacobian[:])

        # print("*** Hessian *** ")
        # print(hessian[2][:])

        # self.outs = outs
        self.jacobian = jacobian
        self.hessian = hessian

    def compute_formula(self,
                        formula,
                        input,
                        input_attr,
                        labels,
                        labels_attr,
                        normal,
                        params=None):

        rst = 0.0

        # print(formula)

        # number of items seperated by add
        if formula.is_Add:
            num_item = len(formula.args)
            # parser each item
            for item in formula.args:
                rst += self.__compute_formula_item(item, input, input_attr,
                                                   labels, labels_attr, normal,
                                                   params)
        else:
            num_item = 1
            rst += self.__compute_formula_item(formula, input, input_attr,
                                               labels, labels_attr, normal,
                                               params)

        return rst

    def __compute_formula_item(self,
                               item,
                               input,
                               input_attr,
                               labels,
                               labels_attr,
                               normal,
                               params=None):

        rst = 1.0  # TODO: float / double / float16

        if item.is_Mul:
            for it in item.args:
                rst = rst * self.__compute_formula_item(
                    it, input, input_attr, labels, labels_attr, normal, params)
        elif item.is_Number:
            # print("*** number:", item)
            rst = float(item) * rst  # TODO: float / double / float16
        elif item.is_Symbol:
            # print("*** symbol:", item)
            rst = rst * self.__compute_formula_symbol(item, input, input_attr)
        elif item.is_Function:
            # print("*** function:", item)
            rst = rst * self.__compute_formula_function(
                item, input, input_attr, labels, labels_attr)
        elif item.is_Derivative:
            # print("*** der:", item)
            rst = rst * self.__compute_formula_der(item, input, normal, params)
        else:
            pass

        return rst

    def __compute_formula_symbol(self, item, input, input_attr):
        var_idx = self.indvar.index(item)
        return self.input[:, var_idx + input_attr.indvar_start]  # TODO

    def __compute_formula_function(self, item, input, input_attr, labels,
                                   labels_attr):

        # output function value
        if item in self.dvar:
            f_idx = self.dvar.index(item)
            return self.outs[:, f_idx]

        # TODO: support u_n as net input
        # # input function value (for time-dependent previous time)
        # if item in self.dvar_n:
        #     f_idx = self.dvar_n.index(item)
        #     return input[:, f_idx + input_attr.dvar_n_start]  # TODO

        if item in self.dvar_n:
            f_idx = self.dvar_n.index(item)
            idx = labels_attr["data_cur"][f_idx]
            return labels[idx]

        # TODO: support parameter pde
        # # parameter pde
        # if item in self.parameter_pde:
        #     f_idx = self.parameter_pde.index(item)
        #     return input[:, f_idx + input_attr.parameter_pde_start]  # TODO

    def __compute_formula_der(self, item, input, normal, params=None):

        jacobian = self.jacobian
        hessian = self.hessian

        # translate sympy diff to f_idx and var_idx
        # f_idx: dependent(function) index, this function is in which index of dependent variable list 
        # var_idx: variable index, this variable is in which index of independent variable list

        # dependent variable
        f_idx = self.dvar.index(item.args[0])

        # derivative order
        order = 0
        for it in item.args[1:]:
            order += it[1]

        # print("  -order: ", order)
        # print("  -f_idx: ", f_idx)

        # parser jacobin for order 1
        if order == 1:
            v = item.args[1][0]
            if v == sympy.Symbol('n'):
                if normal.ndim == 1:
                    rst = paddle.matmul(jacobian[:, f_idx, :], normal)
                else:
                    rst = paddle.dot(normal, jacobian[:, f_idx, :])
            else:
                var_idx = self.indvar.index(v)
                rst = jacobian[:, f_idx, var_idx]

        # parser hessian for order 2
        elif order == 2:
            var_idx = list()
            for it in item.args[1:]:
                for i in range(it[1]):
                    idx = self.indvar.index(it[0])
                    var_idx.append(idx)
            rst = hessian[f_idx][:, var_idx[0], var_idx[1]]

        # order >= 3
        else:
            out = self.outs[:, f_idx]
            for it in item.args[1:]:
                for i in range(it[1]):
                    idx = self.indvar.index(it[0])
                    out = paddle.incubate.autograd.grad(out, input)[:, idx]
            rst = out

        return rst


def l2_norm_square(x, wgt=None):
    if wgt is None:
        return paddle.norm(x**2, p=1)
    else:
        return paddle.norm(x**2 * wgt, p=1)

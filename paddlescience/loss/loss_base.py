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
from paddle.autograd import batch_jacobian, batch_hessian
import sympy
from ..ins import InsDataWithAttr


class LossBase(object):
    def __init__(self, pdes, geo):
        pass

    def eq_loss(self, net):
        pass

    def bc_loss(self, net):
        pass

    def ic_loss(self, net):
        pass


class CompFormula:
    def __init__(self, pde, net):
        self.pde = pde
        self.net = net
        self.order = pde.order
        self.indvar = pde.independent_variable
        self.dvar = pde.dependent_variable
        self.dvar_1 = pde.dependent_variable_1

        self.outs = None
        self.jacobian = None
        self.hessian = None

    def compute_outs_der(self, ins, bs):

        net = self.net

        # outs
        outs = net.nn_func(ins)

        # jacobian
        if self.order >= 1:
            jacobian = batch_jacobian(net.nn_func, ins, create_graph=True)
            jacobian = paddle.reshape(
                jacobian, shape=[net.num_outs, bs, net.num_ins])
        else:
            jacobian = None

        # hessian
        if self.order >= 2:
            for i in range(net.num_outs):

                def func(ins):
                    return net.nn_func(ins)[:, i:i + 1]

                hessian = batch_hessian(func, ins, create_graph=True)
                hessian = paddle.reshape(
                    hessian, shape=[net.num_ins, bs, net.num_ins])
        else:
            hessian = None

        self.outs = outs
        self.jacobian = jacobian
        self.hessian = hessian

    def compute_formula(self, formula, ins, ins_attr, normal):

        # print(formula.args[0])

        rst = 0.0

        # number of items seperated by add
        if formula.is_Add:
            num_item = len(formula.args)
            # parser each item
            for item in formula.args:
                rst += self.__compute_formula_item(item, ins, ins_attr, normal)
        else:
            num_item = 1
            rst += self.__compute_formula_item(formula, ins, ins_attr, normal)

        return rst

    def __compute_formula_item(self, item, ins, ins_attr, normal):

        rst = 1.0  # TODO: float / double / float16

        if item.is_Mul:
            for it in item.args:
                rst = rst * self.__compute_formula_item(it, ins, ins_attr,
                                                        normal)
        elif item.is_Number:
            rst = float(item) * rst  # TODO: float / double / float16
        elif item.is_Symbol:
            #print(item, "symbol")
            rst = rst * self.__compute_formula_symbol(item, ins, ins_attr)
        elif item.is_Function:
            #print(item, "function")
            rst = rst * self.__compute_formula_function(item, ins_attr)
        elif item.is_Derivative:
            # print(item, "der start")
            rst = rst * self.__compute_formula_der(item, normal)
            # print(item, "der end")
        else:
            pass

        return rst

    def __compute_formula_symbol(self, item, ins, ins_attr):
        var_idx = self.indvar.index(item)
        return self.ins[:, var_idx + ins_attr.indvar_start]  # TODO

    def __compute_formula_function(self, item, ins_attr):

        # output function value
        if item in self.dvar:
            f_idx = self.dvar.index(item)
            return self.outs[:, f_idx]

        # input function value (for time-dependent previous time)
        if item in self.dvar_1:
            f_idx = self.dvar_1.index(item)
            return self.ins[:, f_idx + ins_attr.dvar_1_start]  # TODO

        # parameter pde
        if item in self.parameter_pde:
            f_idx = self.parameter_pde.index(item)
            return self.ins[:, f_idx + ins_attr.parameter_pde_start]  # TODO

    def __compute_formula_der(self, item, normal):

        jacobian = self.jacobian
        hessian = self.hessian

        # dependent variable
        f_idx = self.dvar.index(item.args[0])

        # derivative order
        order = 0
        for it in item.args[1:]:
            order += it[1]

        # parser jacobin for order 1
        if order == 1:

            v = item.args[1][0]
            if v == sympy.Symbol('n'):
                rst = normal * jacobian[f_idx, :]
            else:
                var_idx = self.indvar.index(v)
                rst = jacobian[f_idx, :, var_idx]
        # parser hessian for order 2
        elif order == 2:
            if (len(item.args[1:]) == 1):
                var_idx = self.indvar.index(item.args[1][0])
                rst = jacobian[f_idx, :, var_idx]
            else:
                var_idx1 = self.indvar.index(item.args[1][0])
                var_idx2 = self.indvar.index(item.args[2][0])
                rst = hessian[f_idx, var_idx1, :, var_idx2]

        return rst

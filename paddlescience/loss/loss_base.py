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

    def compute_formula(self, formula, ins, normal):

        rst = 0.0

        # number of items seperated by add
        if formula.is_Add:
            num_item = len(formula.args)
            # parser each item
            for item in formula.args:
                rst += self.__compute_formula_item(item, ins, normal)
        else:
            num_item = 1
            rst += self.__compute_formula_item(formula, ins, normal)

        return rst

    def __compute_formula_item(self, item, ins, normal):

        rst = 1.0

        print(item)
        if item.is_Mul:
            for it in item.args:
                rst = rst * self.__compute_formula_item(it, ins, normal)
        elif item.is_Number:
            print(item, "number")
            rst = item * rst
        elif item.is_Symbol:
            print(item, "symbol")
            rst = rst * self.__compute_function_symbol(item, ins)
        elif item.is_Function:
            print(item, "function")
            rst = rst * self.__compute_function_symbol(item)
        elif item.is_Derivative:
            print(item, "der")
            rst = rst * self.__compute_formula_der(item, normal)
            pass
        else:
            pass

        # print(rst)
        return rst

    def __compute_formula_symbol(self, item, ins):
        var_idx = self.indvar.index(item)
        return ins[:, var_idx]

    def __compute_function_symbol(self, item):
        f_idx = self.dvar.index(item)
        return self.outs[:, f_idx]

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

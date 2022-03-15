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


class LossBase(object):
    def __init__(self, pdes, geo):
        pass

    def eq_loss(self, net):
        pass

    def bc_loss(self, net):
        pass

    def ic_loss(self, net):
        pass


class LossDerivative(LossBase):
    def compute_out_der(pde, net, ins, bs):

        # outputs of net
        outs = net.nn_func(ins)

        # jacobian
        if pde.order >= 1:
            jacobian = batch_jacobian(net.nn_func, ins, create_graph=True)
            jacobian = paddle.reshape(
                jacobian, shape=[net.num_outs, bs, net.num_ins])
        else:
            jacobian = None

        # hessian
        if pde.order >= 2:
            for i in range(net.num_outs):

                def func(ins):
                    return net.nn_func(ins)[:, i:i + 1]

                hessian = batch_hessian(func, ins, create_graph=True)
                hessian = paddle.reshape(
                    hessian, shape=[net.num_ins, bs, net.num_ins])
        else:
            hessian = None

        return outs, jacobian, hessian

    def compute_formula(formula, indvar, dvar, ins, outs, jacobian, hessian,
                        normal):

        # number of items seperated by add
        if formula.is_Add:
            num_item = len(formula.args)
        else:
            num_item = 1
        # parser each item
        for item in formula.args:
            #print(item)
            rst = 1.0
            compute_item(indvar, dvar, item, ins, outs, jacobian, hessian, rst)

    def compute_item(indvar, dvar, item, ins, outs, jacobian, hessian, normal,
                     rst):

        #print(item)
        if item.is_Mul:
            for it in item.args:
                rst = compute_item(indvar, dvar, it, ins, outs, jacobian,
                                   hessian, rst)
        elif item.is_Number:
            print(item, "number")
            rst = item * rst
        elif item.is_Symbol:
            print(item, "symbol")
            rst = rst * compute_function(indvar, item, ins)
        elif item.is_Function:
            print(item, "function")
            rst = rst * compute_function(dvar, item, outs)
        elif item.is_Derivative:
            print(item, "der")
            rst = rst * compute_der(indvar, dvar, item, jacobian, hessian,
                                    normal)
            pass
        else:
            pass

        return rst

    def compute_symbol(indvar, item, ins):
        var_idx = indvar.index(item)
        return ins[:, var_idx]

    def compute_function(dvar, item, outs):
        f_idx = dvar.index(item)
        return outs[:, f_idx]

    def compute_der(indvar, dvar, item, jacobian, hessian, normal, rst):

        # dependent variable
        f_idx = dvar.index(item.args[0])

        # derivative order
        order = 0
        for it in item.args[1:]:
            order += it[1]

        # parser jacobin for order 1
        if order == 1:

            v = item.args[1][0]
            if v == sympy.Symbol('n'):
                rst = rst * normal * jacobian[f_idx, :]
            else:
                var_idx = indvar.index(v)
                rst = rst * jacobian[f_idx, var_idx]
        # parser hessian for order 2
        elif order == 2:
            if (len(item.args[1:]) == 1):
                var_idx = indvar.index(item.args[1][0])
                rst = rst * jacobian[f_idx, var_idx]
            else:
                var_idx1 = indvar.index(item.args[1][0])
                var_idx2 = indvar.index(item.args[2][0])
                rst = rst * hessian[f_idx, var_idx1, :, var_idx2]

        return rst

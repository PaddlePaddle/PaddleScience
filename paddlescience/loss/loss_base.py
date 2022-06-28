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
from paddle.autograd import batch_jacobian, batch_hessian
import sympy
from ..inputs import InputsAttr

import time

import jax
from jax import jit
from functools import partial

from .. import config

from paddle.incubate.autograd import Jacobian, Hessian


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
        self.indvar = pde.indvar
        self.dvar = pde.dvar
        self.dvar_n = pde.dvar_n

        self.outs = None
        self.jacobian = None
        self.hessian = None

    def compute_outs(self, input, bs, param=None):
        self.outs = self.net.nn_func(input, param)

    def __func_jac_jax(self, param, x):
        return jax.jacrev(self.net.nn_func, argnums=0)(x, param)

    def __v_func_jac_jax(self, param, x):
        return jax.vmap(self.__func_jac_jax, [None, 0], -1)(param, x)

    # @partial(jit, static_argnums=(0,))
    def __func_hes_jax(self, param, x):
        return jax.hessian(self.net.nn_func, argnums=0)(x, param)

    # @partial(jit, static_argnums=(0,))
    def __v_func_hes_jax(self, param, x):
        return jax.vmap(self.__func_hes_jax, [None, 0], -1)(param, x)

    # def __v_func_hes_jax(self):
    #     # return jax.jit(jax.vmap(self.__func_hes_jax, [None, 0], 0))
    #     return jax.vmap(self.__func_hes_jax, [None, 0], 0)

    def compute_outs_der(self, input, bs, param=None):

        # outs
        self.compute_outs(input, bs, param)

        # jacobian
        if self.order >= 1:

            if config._compute_backend == "jax":

                # def func(param, x):
                #     return jax.jacrev(self.net.nn_func, argnums=0)(x, param)

                # def v_func(param, x):
                #     return jax.vmap(func, [None, 0], -1)(param, x)

                jacobian = self.__v_func_jac_jax(param, input)
            else:
                jacobian = Jacobian(self.net.nn_func, input, is_batched=True)
        else:
            jacobian = None

        # print(jacobian)

        # hessian
        if self.order >= 2:

            # t1 = time.time()

            # hessian = self.__v_func_hes_jax(param, input)

            if config._compute_backend == "jax":
                hessian = self.__v_func_hes_jax(param, input)
            else:
                hessian = list()
                for i in range(self.net.num_outs):

                    def func(input):
                        return self.net.nn_func(input)[:, i:i + 1]

                    hessian.append(Hessian(func, input, is_batched=True))

            # t2 = time.time()
            # print("1: ", t2 - t1)
        else:
            hessian = None

        # self.outs = outs
        self.jacobian = jacobian
        self.hessian = hessian

    def compute_formula(self, formula, input, input_attr, labels, labels_attr,
                        normal):

        rst = 0.0

        # print(formula)

        # number of items seperated by add
        if formula.is_Add:
            num_item = len(formula.args)
            # parser each item
            for item in formula.args:
                rst += self.__compute_formula_item(item, input, input_attr,
                                                   labels, labels_attr, normal)
        else:
            num_item = 1
            rst += self.__compute_formula_item(formula, input, input_attr,
                                               labels, labels_attr, normal)

        return rst

    def __compute_formula_item(self, item, input, input_attr, labels,
                               labels_attr, normal):

        rst = 1.0  # TODO: float / double / float16

        if item.is_Mul:
            for it in item.args:
                rst = rst * self.__compute_formula_item(
                    it, input, input_attr, labels, labels_attr, normal)
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
            rst = rst * self.__compute_formula_der(item, normal)
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

    def __compute_formula_der(self, item, normal):

        jacobian = self.jacobian
        hessian = self.hessian

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
                if api_new:
                    rst = normal * jacobian[:, f_idx, :]  # TODO
                else:
                    rst = normal * jacobian[f_idx, :]
            else:
                var_idx = self.indvar.index(v)
                rst = jacobian[:, f_idx, var_idx]

        # parser hessian for order 2
        elif order == 2:

            # t1 = time.time()

            if (len(item.args[1:]) == 1):
                var_idx = self.indvar.index(item.args[1][0])
                if config._compute_backend == "jax":
                    rst = hessian[f_idx][var_idx][var_idx][:]
                else:
                    rst = hessian[f_idx][:, var_idx, var_idx]
            else:
                var_idx1 = self.indvar.index(item.args[1][0])
                var_idx2 = self.indvar.index(item.args[2][0])

                if config._compute_backend == "jax":
                    rst = hessian[f_idx][var_idx1][var_idx2][:]
                else:
                    rst = hessian[f_idx][:, var_idx1, var_idx2]

            # t2 = time.time()
            # print("2: ", t2-t1)

        return rst

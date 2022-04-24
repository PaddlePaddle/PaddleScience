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
# import paddle.nn.functional as F
# from paddle.autograd import batch_jacobian, batch_hessian
# from ..pde import first_order_rslts, first_order_derivatives, second_order_derivatives
from .loss_base import LossBase, CompFormula
from ..labels import LabelInt


class L2(LossBase):
    """
    L2 loss which is synthesized by three part: the equation loss, the boundary condition loss, and the initial condition loss.

    Parameters:
        pdes(PDE): The partial differential equations used to calculate the equation loss.
        geo(GeometryDiscrete): The discrete geometry on which the loss is calculated.
        aux_func(Callable|None): Optional, default None. If is not None, it should be a python function which returns a list of Paddle Tensors. The list is used as right hand side values when calculating the equation loss.
        eq_weight(float|None): Optional, default None. If is not None, it is multiplied on the equation loss before synthesis.
        bc_weight(numpy.array|None): Optional, default None. If is not None, it should be a 1-D numpy array which has same number of elements as the bodunary condition points. This numpy array is used as weight when calculating the boundary condition loss.
        synthesis_method(string): Optional, default 'add'. The method used when synthesizing the three parts of loss. If is 'add', just add three part directly; If is 'norm', synthesize three part by calculationg 2-norm.
        run_in_batch(bool): Optional, default True. If is True, the equation loss is calculated per batch. If is False, the equation loss is calculated per point.

    Example:
        >>> import paddlescience as psci
        >>> net = psci.loss.L2(pdes=pdes, geo=geo)
    """

    def __init__(self):
        pass

    # compute loss on one interior 
    # there are multiple pde
    def eq_loss(self, pde, net, name_i, input, input_attr, labels, labels_attr,
                bs):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(input, bs)

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

            weight_eq = labels_attr["equations"][i]["weight"]
            if type(weight_eq) == LabelInt:
                weight = labels[weight_eq]
            else:
                weight = weight_eq

            if rhs is None:
                if weight is None:
                    loss += paddle.norm(rst**2, p=1)
                else:
                    loss += paddle.norm(rst**2 * weight, p=1)
            else:
                if weight is None:
                    loss += paddle.norm(rst - rhs, p=2)**2
                else:
                    loss += paddle.norm((rst - rhs)**2 * weight, p=1)

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

            weight_b = labels_attr["bc"][name_b][i]["weight"]

            if type(weight_b) == LabelInt:
                weight = labels[weight_b]
            else:
                weight = weight_b

            if rhs is None:
                if weight is None:
                    loss += paddle.norm(rst, p=2)**2
                else:
                    loss += paddle.norm(rst**2 * weight, p=1)
            else:
                if weight is None:
                    loss += paddle.norm(rst - rhs, p=2)**2
                else:
                    loss += paddle.norm((rst - rhs)**2 * weight, p=1)

        return loss, cmploss.outs

    def ic_loss(self, u, batch_id):
        pass

# compute loss on real data 

    def data_loss(self, pde, net, input, input_attr, labels, labels_attr, bs):

        cmploss = CompFormula(pde, net)

        # compute outs
        cmploss.compute_outs(input, bs)

        loss = 0.0
        for i in range(len(pde.dvar_n)):
            idx = labels_attr["data"][i]
            data = labels[idx]
            loss += paddle.norm(cmploss.outs[:, i] - data, p=2)**2

        return loss, cmploss.outs

    def total_loss(self, input, outs, pde, bs=None):

        # ins_in = ins["interior"]
        # ins_bc = ins["boundary"]

        # lossbc = self.bc_loss(pde, insbc, outsbc, bs)
        # losseq = self.eq_loss(pde, insin, outsin, bs)

        # losstotal = lossbc + losseq

        return losstotal

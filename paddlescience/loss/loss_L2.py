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
import paddle.nn.functional as F
# from paddle.autograd import batch_jacobian, batch_hessian
# from ..pde import first_order_rslts, first_order_derivatives, second_order_derivatives
from .loss_base import LossBase, CompFormula


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

    def eq_loss(self, pde, net, ins, bs):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(ins, bs)

        loss = 0.0
        for formula in pde.equations:
            rst = cmploss.compute_formula(formula, ins, None)
            loss += paddle.norm(rst, p=2)

        return loss, cmploss.outs

    def bc_loss(self, pde, net, ins, bs):

        cmploss = CompFormula(pde, net)

        # compute outs, jacobian, hessian
        cmploss.compute_outs_der(ins, bs)

        loss = 0.0
        for name, bclist in pde.bc.items():
            for b in bclist:
                lhs = cmploss.compute_formula(b.formula, ins,
                                              None)  # TODO: hard code
                rhs = b.rhs  # TODO: to support lambda
                loss += paddle.norm(lhs - rhs, p=2)

        return loss, cmploss.outs

    def ic_loss(self, u, batch_id):
        pass
        # if self.geo.time_dependent == True:
        #     ic_u = paddle.index_select(u, self.geo.ic_index[batch_id])
        #     ic_value = pdes.ic_value
        #     if pdes.ic_check_dim is not None:
        #         ic_u = paddle.index_select(
        #             ic_u, pdes.ic_check_dim, axis=1)
        #     ic_diff = ic_u - ic_value
        #     ic_diff = paddle.reshape(ic_diff, shape=[-1])
        #     return paddle.norm(ic_diff, p=2)
        # else:
        #     return paddle.to_tensor([0], dtype="float32")

    def total_loss(self, ins, outs, pde, bs=None):

        # ins_in = ins["interior"]
        # ins_bc = ins["boundary"]

        # lossbc = self.bc_loss(pde, insbc, outsbc, bs)
        # losseq = self.eq_loss(pde, insin, outsin, bs)

        # losstotal = lossbc + losseq

        return losstotal

        # b_datas = self.geo.get_domain()
        # u = net.nn_func(b_datas)
        # eq_loss = 0
        # if self.run_in_batch:
        #     eq_loss = self.batch_eq_loss(net, b_datas)
        # else:
        #     eq_loss_l = []
        #     for data in b_datas:
        #         eq_loss_l += self.eq_loss(net, data)
        #     eq_loss = paddle.stack(eq_loss_l, axis=0)
        #     eq_loss = paddle.norm(eq_loss, p=2)

        # eq_loss = eq_loss * self.eq_weight if self.eq_weight is not None else eq_loss
        # bc_loss = self.bc_loss(u, batch_id)
        # ic_loss = self.ic_loss(u, batch_id)
        # if self.synthesis_method == 'add':
        #     loss = eq_loss + bc_loss + ic_loss
        #     return loss, [eq_loss, bc_loss, ic_loss]
        # elif self.synthesis_method == 'norm':
        #     losses = [eq_loss, bc_loss, ic_loss]
        #     loss = paddle.norm(paddle.stack(losses, axis=0), p=2)
        #     return loss, losses
        # else:
        #     assert 0, "Unsupported synthesis_method"

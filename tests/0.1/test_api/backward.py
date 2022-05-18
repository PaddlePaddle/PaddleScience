"""
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
"""

import copy
import logging
import numpy as np
import paddle


class DifferenceAppro(object):
    """
    PaddleScience API test base
    """

    def __init__(self, func):
        self.func = func
        self.gap = 1e-3
        self.kwargs = None
        self.var = None
        self.debug = False
        self.hook()

        if self.debug:
            logging.getLogger().setLevel(logging.INFO)

    def hook(self):
        """
        hook
        """
        raise NotImplementedError

    # 计算前向结果
    def cal_rslt(self, **kwargs):
        """
        calculate forward result
        """
        rslt = self.func(**kwargs)
        return rslt

    # 计算反向结果
    def cal_backward(self):
        """
        calculate backward result
        """
        res = self.cal_rslt(**self.kwargs)
        res.backward()
        return self.kwargs["geo"].space_domain.grad

    # 计算一阶导数的差分结果
    def cal_first_derivative(self, **kwargs):
        """
        calculate numerical gradient
        """
        tmp_ins = copy.deepcopy(kwargs["geo"].space_domain)
        bc_index = kwargs["geo"].bc_index
        shape = tmp_ins.shape
        first_numeric_grad = []
        for i in range(len(tmp_ins.flatten())):
            # kwargs = self.kwargs
            tmp1 = tmp_ins.flatten()
            tmp2 = tmp_ins.flatten()
            tmp1[i] += (self.gap / 2)
            tmp2[i] -= (self.gap / 2)
            tmp1 = tmp1.reshape(shape)
            tmp2 = tmp2.reshape(shape)
            kwargs["geo"].space_domain = tmp1
            loss1 = paddle.mean(self.cal_rslt(**kwargs)).numpy()
            kwargs["geo"].bc_index = bc_index
            kwargs["geo"].space_domain = tmp2
            loss2 = paddle.mean(self.cal_rslt(**kwargs)).numpy()
            kwargs["geo"].bc_index = bc_index
            grad = (loss1 - loss2) / self.gap
            first_numeric_grad.append(grad)
        return np.array(first_numeric_grad).reshape(shape)

    def run(self, res=None, **kwargs):
        """
        test run
        """
        logging.info("**************")
        self.kwargs = kwargs
        self.var = kwargs["geo"].space_domain
        if res:
            api_res = self.cal_rslt(**kwargs)
            assert np.allclose(res, api_res), "前向计算错误"
            logging.info("前向计算正确")
        self.compare()

    def compare(self):
        """
        compare backward result
        """
        api_grad = self.cal_backward().numpy()
        # print(api_grad)
        for atol in [1e-4, 1e-3, 1e-2, 1e-1]:
            while self.gap <= 1e-2:
                if not isinstance(self.kwargs["geo"].bc_index, np.ndarray):
                    self.kwargs["geo"].bc_index = self.kwargs["geo"].bc_index[
                        0].numpy()
                self.kwargs["geo"].space_domain = self.var
                numerical_grad = self.cal_first_derivative(**self.kwargs)
                if np.allclose(api_grad, numerical_grad, atol=atol):
                    logging.info("数值梯度精度可模拟，反向计算测试通过")
                    return
                self.gap += 0.001
            self.gap = 1e-3
        assert False, "数值梯度精度不可模拟或反向计算错误"

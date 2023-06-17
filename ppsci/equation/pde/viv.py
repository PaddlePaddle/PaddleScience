# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.nn import initializer

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class Vibration(base.PDE):
    """Vortex induced vibration equation.

    Args:
        rho (float): Generalized mass.
        k1 (float): Learnable paremters for modal damping.
        k2 (float): Learnable paremters for generalized stiffness.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Vibration(1.0, 4.0, -1.0)
    """

    def __init__(self, rho: float, k1: float, k2: float):
        super().__init__()
        self.rho = rho
        self.k1 = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=initializer.Constant(k1),
        )
        self.k2 = paddle.create_parameter(
            shape=[1],
            dtype=paddle.get_default_dtype(),
            default_initializer=initializer.Constant(k2),
        )
        self.learnable_parameters.append(self.k1)
        self.learnable_parameters.append(self.k2)

        def f_compute_func(out):
            eta, t = out["eta"], out["t_f"]
            eta__t = jacobian(eta, t)
            eta__t__t = hessian(eta, t)
            f = (
                self.rho * eta__t__t
                + paddle.exp(self.k1) * eta__t
                + paddle.exp(self.k2) * eta
            )
            return f

        self.add_equation("f", f_compute_func)

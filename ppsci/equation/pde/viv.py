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

from __future__ import annotations

import paddle
import sympy as sp
from paddle.nn import initializer

from ppsci.equation.pde import base


class Vibration(base.PDE):
    r"""Vortex induced vibration equation.

    $$
    \rho \dfrac{\partial^2 \eta}{\partial t^2} + e^{k1} \dfrac{\partial \eta}{\partial t} + e^{k2} \eta = f
    $$

    Args:
        rho (float): Generalized mass.
        k1 (float): Learnable parameter for modal damping.
        k2 (float): Learnable parameter for generalized stiffness.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Vibration(1.0, 4.0, -1.0)
    """

    def __init__(self, rho: float, k1: float, k2: float):
        super().__init__()
        self.rho = rho
        self.k1 = paddle.create_parameter(
            shape=[],
            dtype=paddle.get_default_dtype(),
            default_initializer=initializer.Constant(k1),
        )
        self.k2 = paddle.create_parameter(
            shape=[],
            dtype=paddle.get_default_dtype(),
            default_initializer=initializer.Constant(k2),
        )
        self.learnable_parameters.append(self.k1)
        self.learnable_parameters.append(self.k2)

        t_f = self.create_symbols("t_f")
        eta = self.create_function("eta", (t_f,))
        k1 = self.create_symbols(self.k1.name)
        k2 = self.create_symbols(self.k2.name)
        f = self.rho * eta.diff(t_f, 2) + sp.exp(k1) * eta.diff(t_f) + sp.exp(k2) * eta
        self.add_equation("f", f)

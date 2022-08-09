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

from .pde_base import PDE
import sympy
import numpy as np

__all__ = ['PlateEquilibrium']


# PlateEquilibrium equation
class PlateEquilibrium(PDE):
    def __init__(self,
                 stiff,
                 mass=1.0,
                 rhs=None,
                 time_dependent=False,
                 weight=1.0):

        if time_dependent == True:

            t = sympy.Symbol('t')
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            w = sympy.Function('w')(t, x, y)
            super(PlateEquilibrium, self).__init__([t, x, y], [w], weight)

            w4x = w.diff(x).diff(x).diff(x).diff(x)
            w4y = w.diff(y).diff(y).diff(y).diff(y)
            w2x2y = w.diff(y).diff(y).diff(x).diff(x)
            w2t = w.diff(t).diff(t)

            eq = stiff * w4x + 2.0 * stiff * w2x2y + stiff * w4y + mass * w.diff(
                t).diff(t)
            self.add_equation(eq, rhs)

        else:

            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            w = sympy.Function('w')(x, y)
            super(PlateEquilibrium, self).__init__([x, y], [w], weight)

            w4x = w.diff(x).diff(x).diff(x).diff(x)
            w4y = w.diff(y).diff(y).diff(y).diff(y)
            w2x2y = w.diff(y).diff(y).diff(x).diff(x)

            eq = stiff * w4x + 2.0 * stiff * w2x2y + stiff * w4y
            self.add_equation(eq, rhs)

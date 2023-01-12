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

from .. import config
from .pde_base import PDE
from ..parameter import Parameter, is_parameter

import sympy
import numpy as np

__all__ = ['EulerBeam']


class EulerBeam(PDE):
    """
    beam wrap equation

    Parameters:
        E (float): elastic modulus
        q (float): load density
        dim (integer): dquation's dimention. 1 and 2 are supported.
        time_dependent (bool): time-dependent or time-independent.
        weight (optional, float or list of float or lambda function): weight in computing equation loss. The default value is 1.0.        

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.pde.EulerBeam(E=1.0, q=1.0, dim=1)
    """

    def __init__(self,
                 E=1.0,
                 q=1.0,
                 mass=1.0,
                 rhs=None,
                 time_dependent=False,
                 weight=None):


            x = sympy.Symbol('x')
            u = sympy.Function('u')(x)
            u_x = sympy.Function('u_x')(x)
            u_xx = sympy.Function('u_xx')(x)
            u_xxx = sympy.Function('u_xxx')(x)
            
            super(EulerBeam, self).__init__([x], [u], weight)

            u_x = u.diff(x)
            u_xx = u_x.diff(x)
            u_xxx = u_xx.diff(x)
            u_xxxx = u_xxx.diff(x)
            # u_t = u.diff(t)

            eq = E * u_xxxx + 1
            self.add_equation(eq, rhs)   


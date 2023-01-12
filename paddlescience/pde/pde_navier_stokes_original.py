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

__all__ = ['NavierStokes']


class NavierStokes(PDE):
    """
    Navier-Stokes equation

    .. math::
        :nowrap:

        Time-independent Navier-Stokes Equation

        \\begin{eqnarray*}
            && \\frac{\\partial u}{\\partial x} + \\frac{\\partial v}{\\partial y} + \\frac{\\partial w}{\\partial z} = 0,   \\\\
            && u \\frac{\\partial u}{\\partial x} +  v \\frac{\partial u}{\\partial y} +  w \\frac{\partial u}{\\partial z} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 u}{\\partial x^2} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 u}{\\partial z^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 u}{\\partial y^2} + \\frac{\\partial p}{\\partial x} = 0,\\\\
            && u \\frac{\\partial v}{\\partial x} +  v \\frac{\partial v}{\\partial y} +  w \\frac{\partial v}{\\partial z} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 v}{\\partial x^2} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 v}{\\partial z^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 v}{\\partial y^2} + \\frac{\\partial p}{\\partial y}  = 0, \\\\
            && u \\frac{\\partial w}{\\partial x} +  v \\frac{\partial w}{\\partial y} +  w \\frac{\partial w}{\\partial z} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 w}{\\partial x^2} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 w}{\\partial z^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 w}{\\partial y^2} + \\frac{\\partial p}{\\partial z}  = 0.
        \\end{eqnarray*}

        Time-dependent Navier-Stokes equation

        \\begin{eqnarray*}
            && \\frac{\\partial u}{\\partial x} + \\frac{\\partial v}{\\partial y} + \\frac{\\partial w}{\\partial z} = 0,   \\\\
            && \\frac{\\partial u}{\\partial t} + u \\frac{\\partial u}{\\partial x} +  v \\frac{\partial u}{\\partial y} +  w \\frac{\partial u}{\\partial z} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 u}{\\partial x^2} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 u}{\\partial z^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 u}{\\partial y^2} + \\frac{\\partial p}{\\partial x} = 0,\\\\
            && \\frac{\\partial v}{\\partial t} + u \\frac{\\partial v}{\\partial x} +  v \\frac{\partial v}{\\partial y} +  w \\frac{\partial v}{\\partial z} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 v}{\\partial x^2} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 v}{\\partial z^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 v}{\\partial y^2} + \\frac{\\partial p}{\\partial y} = 0, \\\\
            && \\frac{\\partial w}{\\partial t} + u \\frac{\\partial w}{\\partial x} +  v \\frac{\partial w}{\\partial y} +  w \\frac{\partial w}{\\partial z} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 w}{\\partial x^2} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 w}{\\partial z^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 w}{\\partial y^2} + \\frac{\\partial p}{\\partial z} = 0.
        \\end{eqnarray*}

    Parameters:
        nu (float): kinematic viscosity.
        rho (float): density.
        dim (integer): dquation's dimention. 2 and 3 are supported.
        time_dependent (bool): time-dependent or time-independent.
        weight (optional, float or list of float or lambda function): weight in computing equation loss. The default value is 1.0.        

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2)
    """

    def __init__(self,
                 nu=0.01,
                 rho=1.0,
                 dim=2,
                 time_dependent=False,
                 weight=None):

        if dim == 2 and time_dependent == False:

            # independent and dependent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            u = sympy.Function('u')(x, y)
            v = sympy.Function('v')(x, y)
            p = sympy.Function('p')(x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y)
            continuty_rhs = 0
            # momentum equation
            momentum_x = u * u.diff(x) + v * u.diff(y) - nu / rho * u.diff(
                x).diff(x) - nu / rho * u.diff(y).diff(y) + 1.0 / rho * p.diff(
                    x)
            momentum_y = u * v.diff(x) + v * v.diff(y) - nu / rho * v.diff(
                x).diff(x) - nu / rho * v.diff(y).diff(y) + 1.0 / rho * p.diff(
                    y)
            momentum_x_rhs = 0
            momentum_y_rhs = 0

            super(NavierStokes, self).__init__([x, y], [u, v, p], weight)
            self.add_equation(continuty, continuty_rhs)
            self.add_equation(momentum_x, momentum_x_rhs)
            self.add_equation(momentum_y, momentum_y_rhs)

        elif dim == 2 and time_dependent == True:

            # independent variable
            t = sympy.Symbol('t')
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable
            u = sympy.Function('u')(t, x, y)
            v = sympy.Function('v')(t, x, y)
            p = sympy.Function('p')(t, x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y)
            continuty_rhs = 0

            # momentum x equation
            momentum_x = u.diff(t) + u * u.diff(x) + v * u.diff(
                y) - nu / rho * u.diff(x).diff(x) - nu / rho * u.diff(y).diff(
                    y) + 1.0 / rho * p.diff(x)
            momentum_y = v.diff(t) + u * v.diff(x) + v * v.diff(
                y) - nu / rho * v.diff(x).diff(x) - nu / rho * v.diff(y).diff(
                    y) + 1.0 / rho * p.diff(y)
            momentum_x_rhs = 0
            momentum_y_rhs = 0

            super(NavierStokes, self).__init__([t, x, y], [u, v, p], weight)
            self.add_equation(continuty, continuty_rhs)
            self.add_equation(momentum_x, momentum_x_rhs)
            self.add_equation(momentum_y, momentum_y_rhs)

        elif dim == 3 and time_dependent == False:

            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            # dependent variable
            u = sympy.Function('u')(x, y, z)
            v = sympy.Function('v')(x, y, z)
            w = sympy.Function('w')(x, y, z)
            p = sympy.Function('p')(x, y, z)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y) + w.diff(z)
            continuty_rhs = 0

            # momentum x equation
            momentum_x = u * u.diff(x) + v * u.diff(y) + w * u.diff(
                z) - nu / rho * u.diff(x).diff(x) - nu / rho * u.diff(y).diff(
                    y) - nu / rho * u.diff(z).diff(z) + 1.0 / rho * p.diff(x)
            momentum_y = u * v.diff(x) + v * v.diff(y) + w * v.diff(
                z) - nu / rho * v.diff(x).diff(x) - nu / rho * v.diff(y).diff(
                    y) - nu / rho * v.diff(z).diff(z) + 1.0 / rho * p.diff(y)
            momentum_z = u * w.diff(x) + v * w.diff(y) + w * w.diff(
                z) - nu / rho * w.diff(x).diff(x) - nu / rho * w.diff(y).diff(
                    y) - nu / rho * w.diff(z).diff(z) + 1.0 / rho * p.diff(z)
            momentum_x_rhs = 0
            momentum_y_rhs = 0
            momentum_z_rhs = 0

            super(NavierStokes, self).__init__([x, y, z], [u, v, w, p], weight)
            self.add_equation(continuty, continuty_rhs)
            self.add_equation(momentum_x, momentum_x_rhs)
            self.add_equation(momentum_y, momentum_y_rhs)
            self.add_equation(momentum_z, momentum_z_rhs)

        elif dim == 3 and time_dependent == True:

            # independent variable
            t = sympy.Symbol('t')
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            # dependent variable
            u = sympy.Function('u')(t, x, y, z)
            v = sympy.Function('v')(t, x, y, z)
            w = sympy.Function('w')(t, x, y, z)
            p = sympy.Function('p')(t, x, y, z)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y) + w.diff(z)
            continuty_rhs = 0

            # momentum x equation
            momentum_x = u.diff(t) + u * u.diff(x) + v * u.diff(
                y) + w * u.diff(z) - nu / rho * u.diff(x).diff(
                    x) - nu / rho * u.diff(y).diff(y) - nu / rho * u.diff(
                        z).diff(z) + 1.0 / rho * p.diff(x)
            momentum_y = v.diff(t) + u * v.diff(x) + v * v.diff(
                y) + w * v.diff(z) - nu / rho * v.diff(x).diff(
                    x) - nu / rho * v.diff(y).diff(y) - nu / rho * v.diff(
                        z).diff(z) + 1.0 / rho * p.diff(y)
            momentum_z = w.diff(t) + u * w.diff(x) + v * w.diff(
                y) + w * w.diff(z) - nu / rho * w.diff(x).diff(
                    x) - nu / rho * w.diff(y).diff(y) - nu / rho * w.diff(
                        z).diff(z) + 1.0 / rho * p.diff(z)
            momentum_x_rhs = 0
            momentum_y_rhs = 0
            momentum_z_rhs = 0

            super(NavierStokes, self).__init__([t, x, y, z], [u, v, w, p],
                                               weight)
            self.add_equation(continuty, continuty_rhs)
            self.add_equation(momentum_x, momentum_x_rhs)
            self.add_equation(momentum_y, momentum_y_rhs)
            self.add_equation(momentum_z, momentum_z_rhs)

        # parameter list
        self.nu = nu
        self.rho = rho
        if is_parameter(nu):
            self.parameter.append(nu)
        if is_parameter(rho):
            self.parameter.append(rho)

        self.dim = dim

    def time_discretize(self, time_method=None, time_step=None):
        if time_method is None:
            pde_disc = self
        elif time_method == "implicit":
            pde_disc = NavierStokesImplicit(
                nu=self.nu,
                rho=self.rho,
                dim=self.dim,
                time_step=time_step,
                weight=self.weight)
        else:
            pass
            # TODO: error out

        return pde_disc


class NavierStokesImplicit(PDE):
    def __init__(self, nu=0.01, rho=1.0, dim=2, time_step=None, weight=None):

        # time step
        self.dt = time_step
        dt = time_step

        if dim == 2:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable current time step: u^{n+1}, v^{n+1}, p^{n+1}
            u = sympy.Function('u')(x, y)
            v = sympy.Function('v')(x, y)
            p = sympy.Function('p')(x, y)

            # dependent variable previous time step: u^{n}, v^{n}, p^{n}
            u_n = sympy.Function('u_n')(x, y)
            v_n = sympy.Function('v_n')(x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y)
            continuty_rhs = 0

            # momentum
            momentum_x = u / dt - u_n / dt + u * u.diff(x) + v * u.diff(
                y) - nu / rho * u.diff(x).diff(x) - nu / rho * u.diff(y).diff(
                    y) + 1.0 / rho * p.diff(x)
            momentum_y = v / dt - v_n / dt + u * v.diff(x) + v * v.diff(
                y) - nu / rho * v.diff(x).diff(x) - nu / rho * v.diff(y).diff(
                    y) + 1.0 / rho * p.diff(y)
            momentum_x_rhs = 0
            momentum_y_rhs = 0

            super(NavierStokesImplicit, self).__init__([x, y], [u, v, p],
                                                       weight)
            self.dvar_n = [u_n, v_n]
            self.add_equation(continuty, continuty_rhs)
            self.add_equation(momentum_x, momentum_x_rhs)
            self.add_equation(momentum_y, momentum_y_rhs)

        elif dim == 3:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            # dependent variable current time step: u^{n+1}, v^{n+1}, w^{n+1}, p^{n+1}
            u = sympy.Function('u')(x, y, z)
            v = sympy.Function('v')(x, y, z)
            w = sympy.Function('w')(x, y, z)
            p = sympy.Function('p')(x, y, z)

            # dependent variable previous time step: u^{n}, v^{n}, w^{n}
            u_n = sympy.Function('u_n')(x, y, z)
            v_n = sympy.Function('v_n')(x, y, z)
            w_n = sympy.Function('w_n')(x, y, z)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y) + w.diff(z)
            continuty_rhs = 0

            # momentum
            momentum_x = u / dt - u_n / dt + u * u.diff(x) + v * u.diff(
                y) + w * u.diff(z) - nu / rho * u.diff(x).diff(
                    x) - nu / rho * u.diff(y).diff(y) - nu / rho * u.diff(
                        z).diff(z) + 1.0 / rho * p.diff(x)
            momentum_y = v / dt - v_n / dt + u * v.diff(x) + v * v.diff(
                y) + w * v.diff(z) - nu / rho * v.diff(x).diff(
                    x) - nu / rho * v.diff(y).diff(y) - nu / rho * v.diff(
                        z).diff(z) + 1.0 / rho * p.diff(y)
            momentum_z = w / dt - w_n / dt + u * w.diff(x) + v * w.diff(
                y) + w * w.diff(z) - nu / rho * w.diff(x).diff(
                    x) - nu / rho * w.diff(y).diff(y) - nu / rho * w.diff(
                        z).diff(z) + 1.0 / rho * p.diff(z)
            momentum_x_rhs = 0
            momentum_y_rhs = 0
            momentum_z_rhs = 0

            super(NavierStokesImplicit, self).__init__([x, y, z],
                                                       [u, v, w, p], weight)
            self.dvar_n = [u_n, v_n, w_n]
            self.add_equation(continuty, continuty_rhs)
            self.add_equation(momentum_x, momentum_x_rhs)
            self.add_equation(momentum_y, momentum_y_rhs)
            self.add_equation(momentum_z, momentum_z_rhs)

        # parameter list
        self.time_dependent = True
        self.time_disc_method = "implicit"

        if is_parameter(nu):
            self.parameter.append(nu)
        if is_parameter(rho):
            self.parameter.append(rho)

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

from .pde_base import PDE
from ..parameter import Parameter, is_parameter

import sympy
import numpy as np


#class NavierStokes:
class NavierStokes(PDE):
    """
    Two dimentional time-independent Navier-Stokes equation  

    .. math::
        :nowrap:

        \\begin{eqnarray*}
            \\frac{\\partial u}{\\partial x} + \\frac{\\partial u}{\\partial y} & = & 0,   \\\\
            u \\frac{\\partial u}{\\partial x} +  v \\frac{\partial u}{\\partial y} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 u}{\\partial x^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 u}{\\partial y^2} + dp/dx & = & 0,\\\\
            u \\frac{\\partial v}{\\partial x} +  v \\frac{\partial v}{\\partial y} - \\frac{\\nu}{\\rho} \\frac{\\partial^2 v}{\\partial x^2} - \\frac{\\nu}{\\rho}  \\frac{\\partial^2 v}{\\partial y^2} + dp/dy & = & 0.
        \\end{eqnarray*}

    Parameters
    ----------
        nu : float
            Kinematic viscosity
        rho : float
            Density

    Example:
        >>> import paddlescience as psci
        >>> pde = psci.pde.NavierStokes(0.01, 1.0)
    """

    def __init__(self,
                 nu=0.01,
                 rho=1.0,
                 dim=2,
                 time_dependent=False,
                 weight=None):
        super(NavierStokes, self).__init__(
            dim + 1, time_dependent=time_dependent, weight=weight)

        # parameter list
        if is_parameter(nu):
            self.parameter.append(nu)
        if is_parameter(rho):
            self.parameter.append(rho)

        if dim == 2 and time_dependent == False:

            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable
            u = sympy.Function('u')(x, y)
            v = sympy.Function('v')(x, y)
            p = sympy.Function('p')(x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(y)
            continuty_rhs = 0

            # momentum x equation
            momentum_x = u * u.diff(x) + v * u.diff(y) - nu / rho * u.diff(
                x).diff(x) - nu / rho * u.diff(y).diff(y) + 1.0 / rho * p.diff(
                    x)
            momentum_y = u * v.diff(x) + v * v.diff(y) - nu / rho * v.diff(
                x).diff(x) - nu / rho * v.diff(y).diff(y) + 1.0 / rho * p.diff(
                    y)
            momentum_x_rhs = 0
            momentum_y_rhs = 0

            # variables in order
            self.independent_variable = [x, y]
            self.dependent_variable = [u, v, p]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.rhs = list()
            self.equations.append(continuty)
            self.equations.append(momentum_x)
            self.equations.append(momentum_y)
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)
            self.rhs.append(momentum_y_rhs)

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

            # variables in order
            self.independent_variable = [t, x, y]
            self.dependent_variable = [u, v, p]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.rhs = list()
            self.equations.append(continuty)
            self.equations.append(momentum_x)
            self.equations.append(momentum_y)
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)
            self.rhs.append(momentum_y_rhs)

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

            # variables in order
            self.independent_variable = [x, y, z]
            self.dependent_variable = [u, v, w, p]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.rhs = list()
            self.equations.append(continuty)
            self.equations.append(momentum_x)
            self.equations.append(momentum_y)
            self.equations.append(momentum_z)
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)
            self.rhs.append(momentum_y_rhs)
            self.rhs.append(momentum_z_rhs)

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

            # variables in order
            self.independent_variable = [t, x, y, z]
            self.dependent_variable = [u, v, w, p]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.rhs = list()
            self.equations.append(continuty)
            self.equations.append(momentum_x)
            self.equations.append(momentum_y)
            self.equations.append(momentum_z)
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)
            self.rhs.append(momentum_y_rhs)
            self.rhs.append(momentum_z_rhs)

    def discretize(self, dt=1.0):

        if self.time_dependent == False:
            return self
        else:
            return NavierStokesImplicit(self, dt)

    def discretize_bc(self, geometry):

        # discritize weight and rhs in boundary condition
        for name_b, bc in self.bc.items():
            points_b = geometry.boundary[name_b]

            data = list()
            for n in range(len(points_b[0])):
                data.append(points_b[:, n])

            # boundary weight
            for b in bc:
                # compute weight lambda with cordinates
                if type(b.weight) == types.LambdaType:
                    b.weight_disc = b.weight(*data)
                else:
                    b.weight_disc = b.weight

            # boundary rhs
            for b in bc:
                if type(b.rhs) == types.LambdaType:
                    b.rhs_disc = b.rhs(*data)
                else:
                    b.rhs_disc = b.rhs

    def discretize_geometry(self):
        pass


class NavierStokesImplicit(PDE):
    def __init__(self, ns):
        super(NavierStokesImplicit, self).__init__(dim + 1, dt)

        self.time_dependent = True
        self.time_discretize_method = "implicit"
        self.nu = ns.nu
        self.rho = ns.rho
        self.bc = ns.bc
        self.dt = dt

        if dim == 2:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')

            # dependent variable current time step: u^{n}, v^{n}, p^{n}
            u = sympy.Function('u')(x, y)
            v = sympy.Function('v')(x, y)
            p = sympy.Function('p')(x, y)

            # dependent variable previous time step: u^{n-1}, v^{n-1}, p^{n-1}
            u_1 = sympy.Function('u_1')(x, y)
            v_1 = sympy.Function('v_1')(x, y)
            p_1 = sympy.Function('p_1')(x, y)

            # normal direction
            self.normal = sympy.Symbol('n')

            # continuty equation
            continuty = u.diff(x) + v.diff(x)
            continuty_rhs = 0

            # momentum
            momentum_x = u / dt - u_1 / dt + u * u.diff(x) + v * u.diff(
                y) - nu / rho * u.diff(x).diff(x) - nu / rho * u.diff(y).diff(
                    y) + 1.0 / rho * p.diff(x)
            momentum_y = v / dt - v_1 / dt + u * v.diff(x) + v * v.diff(
                y) - nu / rho * v.diff(x).diff(x) - nu / rho * v.diff(y).diff(
                    y) + 1.0 / rho * p.diff(y)
            momentum_x_rhs = 0
            momentum_y_rhs = 0

            # variables in order
            self.independent_variable = [x, y]
            self.dependent_variable = [u, v, p]
            self.dependent_variable_1 = [u_1, v_1, p_1]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.rhs = list()
            self.equations.append(continuty)
            self.equations.append(momentum_x)
            self.equations.append(momentum_y)
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)
            self.rhs.append(momentum_y_rhs)

        elif dim == 3:
            # independent variable
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            # dependent variable current time step: u^{n}, v^{n}, w^{n}, p^{n}
            u = sympy.Function('u')(x, y, z)
            v = sympy.Function('v')(x, y, z)
            w = sympy.Function('w')(x, y, z)
            p = sympy.Function('p')(x, y, z)

            # dependent variable previous time step: u^{n-1}, v^{n-1}, w^{n-1}, p^{n-1}
            u_1 = sympy.Function('u_1')(x, y, z)
            v_1 = sympy.Function('v_1')(x, y, z)
            w_1 = sympy.Function('w_1')(x, y, z)
            p_1 = sympy.Function('p_1')(x, y, z)

            # normal direction
            self.normal = sympy.Symbol('n')

            # dt
            self.dt = sympy.Symbol('dt')

            # continuty equation
            continuty = u.diff(x) + v.diff(x) + w.diff(z)
            continuty_rhs = 0

            # momentum
            momentum_x = u / dt - u_1 / dt + u * u.diff(x) + v * u.diff(
                y) + w * u.diff(z) - nu / rho * u.diff(x).diff(
                    x) - nu / rho * u.diff(y).diff(y) - nu / rho * u.diff(
                        z).diff(z) + 1.0 / rho * p.diff(x)
            momentum_y = v / dt - v_1 / dt + u * v.diff(x) + v * v.diff(
                y) + w * v.diff(z) - nu / rho * v.diff(x).diff(
                    x) - nu / rho * v.diff(y).diff(y) - nu / rho * v.diff(
                        z).diff(z) + 1.0 / rho * p.diff(y)
            momentum_z = w / dt - w_1 / dt + u * w.diff(x) + v * w.diff(
                y) + w * w.diff(z) - nu / rho * w.diff(x).diff(
                    x) - nu / rho * w.diff(y).diff(y) - nu / rho * w.diff(
                        z).diff(z) + 1.0 / rho * p.diff(z)
            momentum_x_rhs = 0
            momentum_y_rhs = 0
            momentum_z_rhs = 0

            # variables in order
            self.independent_variable = [x, y, z]
            self.dependent_variable = [u, v, w, p]
            self.dependent_variable_1 = [u_1, v_1, w_1, p_1]

            # order
            self.order = 2

            # equations and rhs
            self.equations = list()
            self.rhs = list()
            self.equations.append(continuty)
            self.equations.append(momentum_x)
            self.equations.append(momentum_y)
            self.equations.append(momentum_z)
            self.rhs.append(continuty_rhs)
            self.rhs.append(momentum_x_rhs)
            self.rhs.append(momentum_y_rhs)
            self.rhs.append(momentum_z_rhs)

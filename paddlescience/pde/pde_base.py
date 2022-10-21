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
import numpy as np
import types
import sympy
import collections

__all__ = ['PDE']


class PDE:
    """
    User-define Equation

    This module supports to define equations with following steps

        1. define independent variable
        2. define dependent variable
        3. declare PDE defining independent variable and dependent variable
        4. define equations and rhs

    Example:
        >>> # Example: 2-dimentional Laplace equation
        >>> x = sympy.Symbol('x')
        >>> y = sympy.Symbol('y')
        >>> u = sympy.Function('u')(x,y)
        >>> pde = psci.pde.PDE(indvar=[x, y], dvar=[u])
        >>> pde.add_equation(u.diff(x).diff(x) + u.diff(y).diff(y))
    """

    def __init__(self, indvar, dvar, weight=None):

        # independent variable
        if type(indvar) is list or type(indvar) is tuple:
            self.indvar = indvar
        else:
            pass  # TODO: error out

        # dependent variable on current time step n
        if type(dvar) is list or type(dvar) is tuple:
            self.dvar = dvar
        else:
            pass  # TODO: error out

        # dependent variable on previous time step n-1
        self.dvar_n = list()

        # time dependent / time independent
        self.time_dependent = False
        for var in self.indvar:
            if var == sympy.Symbol('t'):
                self.time_dependent = True
                break

        # parameter in pde
        self.parameter = list()

        # equation and right-hand side
        self.equations = list()
        self.rhs = list()

        # # order
        # self.order = order

        # boundary condition
        self.bc = collections.OrderedDict()

        # initial condition
        self.ic = list()

        # geometry
        self.geometry = None

        # weight
        self.weight = weight

        # rhs disc
        self.rhs_disc = list()

        # weight disc
        self.weight_disc = list()

        # discretize method (for time-dependent)
        self.time_disc_method = None

        # time interval
        self.time_internal = None
        self.time_step = None
        self.time_array = None

    def add_geometry(self, geo):
        self.geometry = geo

    def add_equation(self, eq, rhs=0.0):
        """
        Add equation to PDE

        Parameters:
            eq: equation defined with python sympy module.

        Example:
            >>> import paddlescience as psci
            >>> x = sympy.Symbol('x')
            >>> y = sympy.Symbol('y')
            >>> u = sympy.Function('u')(x,y)
            >>> pde = psci.pde.PDE(indvar=[x,y], dvar=[u])
            >>> pde.add_equation(u.diff(x).diff(x) + u.diff(y).diff(y))
        """
        self.equations.append(eq)
        self.rhs.append(rhs)

    def set_bc(self, name, *args):
        """
        Add boundary condition to boundary

        Parameters:
            name (string): boundary name.
            args (boundary conditions): boundary conditions which are added to boundary. 

        Example:
            >>> import paddlescience as psci
            >>> geo = psci.geometry.Rectangular(origin=(0.0, 0.0), extent=(1.0, 1.0))
            >>> geo.add_boundary(name="top",criteria=lambda x, y: (y == 1.0))
            >>> pde = psci.pde.Laplace(dim=2)
            >>> bc1 = psci.bc.Dirichlet('u', rhs=0)
            >>> bc2 = psci.bc.Dirichlet('v', rhs=0)
            >>> pde.set_bc("top", bc1, bc2) # add boundary conditions to boundary "top"
        """

        if name not in self.bc:
            self.bc[name] = list()

        for arg in args:
            arg.to_formula(self.indvar)
            self.bc[name].append(arg)

    def add_bc(self, name, *args):
        self.set_bc(name, *args)

    def set_ic(self, *args):
        """
        Add initial condition for time-dependent equation

        Parameters:
            args (initial conditions): initial conditions 

        Example:
            >>> pde = psci.pde.NavierStokes(dim=3, time_dependent=True)
            >>> ic1 = psci.ic.IC('u', rhs=0) 
            >>> ic2 = psci.ic.IC('v', rhs=0) 
            >>> pde.set_ic(ic1, ic2)         # add initial conditions
       """
        for arg in args:
            arg.to_formula(self.indvar)
            self.ic.append(arg)

    def add_ic(self, *args):
        self.set_ic(*args)

    def set_time_interval(self, interval):
        """
        Set time interval (start time and end time) for time-dependent equation

        Parameters:
            interval: [start time, end time]

        Example:
            >>> import paddlescience as psci
            >>> pde = psci.pde.NavierStokes(dim=2, time_dependent=True)
            >>> pde.set_time_interval([0.0, 1.0]) # time interval [0.0, 1.0]
        """
        self.time_internal = interval

    def discretize(self,
                   time_method=None,
                   time_step=None,
                   time_array=None,
                   geo_disc=None):
        """
        Discretize equations

        Parameters:
            time_method (None or "implicit"): 
                "implicit": discretize time-dependent Navier-Stokes equations with implicit method
            time_step (integer): number of time steps for time-dependent equation
            geo_disc (GeometryDisc): discrete geometry

        Example:
            >>> import paddlescience as psci
            >>> pde = psci.pde.NavierStokes(dim=3, time_dependent=True)
            >>> pde.set_time_interval([0.0, 1.0]) # time interval [0.0, 1.0]
            >>> pde.discretize(geo_disc=geo_disc)
        """

        # time discretize pde
        if self.time_dependent:
            pde_disc = self.time_discretize(time_method, time_step)

            # time interval
            if time_array is None:
                pde_disc.time_internal = self.time_internal
                pde_disc.time_step = time_step
                t0 = self.time_internal[0]
                t1 = self.time_internal[1]
                n = int((t1 - t0) / time_step) + 1
                pde_disc.time_array = np.linspace(
                    t0, t1, n, dtype=config._dtype)
            else:
                pde_disc.time_array = time_array.astype(config._dtype)
        else:
            pde_disc = self

        # geometry
        pde_disc.geometry = geo_disc

        # bc
        for name, bc in self.bc.items():
            pde_disc.bc[name] = list()
            for i in range(len(bc)):
                bc_disc = bc[i].discretize(pde_disc.indvar)
                pde_disc.bc[name].append(bc_disc)

        # discritize r hs in equation for interior points
        pde_disc.rhs_disc = dict()
        pde_disc.rhs_disc["interior"] = list()
        for rhs in pde_disc.rhs:
            points_i = pde_disc.geometry.interior

            data = list()
            for n in range(len(points_i[0])):
                data.append(points_i[:, n])

            if type(rhs) == types.LambdaType:
                pde_disc.rhs_disc["interior"].append(rhs(*data))
            else:
                pde_disc.rhs_disc["interior"].append(rhs)

        # discritize rhs in equation for user points
        if pde_disc.geometry.user is not None:
            pde_disc.rhs_disc["user"] = list()
            for rhs in pde_disc.rhs:
                points_i = pde_disc.geometry.user

                data = list()
                for n in range(len(points_i[0])):
                    data.append(points_i[:, n])

                if type(rhs) == types.LambdaType:
                    pde_disc.rhs_disc["user"].append(rhs(*data))
                else:
                    pde_disc.rhs_disc["user"].append(rhs)

        # discretize weight in equations
        weight = pde_disc.weight
        if (weight is None) or np.isscalar(weight):
            pde_disc.weight_disc = [weight for _ in range(len(self.equations))]
        else:
            pde_disc.weight_disc = weight
            # TODO: points dependent value

        # discritize weight and rhs in boundary condition
        for name_b, bc in pde_disc.bc.items():
            points_b = pde_disc.geometry.boundary[name_b]
            normal_b = pde_disc.geometry.normal[name_b]

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

            # boundary normal
            for b in bc:
                b.normal_disc = normal_b

        # discretize rhs in initial condition
        for ic in pde_disc.ic:
            points_i = pde_disc.geometry.interior

            data = list()
            for n in range(len(points_i[0])):
                data.append(points_i[:, n])

            rhs = ic.rhs
            if type(rhs) == types.LambdaType:
                ic.rhs_disc = rhs(*data)
            else:
                ic.rhs_disc = rhs

        return pde_disc

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

    def __init__(self, nu=0.01, rho=1.0, dim=2, time_dependent=False):
        super(NavierStokes, self).__init__(
            dim + 1, time_dependent=time_dependent)
        if dim == 2 and time_dependent == False:
            # continuty 
            self.add_item(0, 1.0, "du/dx")
            self.add_item(0, 1.0, "dv/dy")
            # momentum x
            self.add_item(1, 1.0, "u", "du/dx")
            self.add_item(1, 1.0, "v", "du/dy")
            self.add_item(1, -nu / rho, "d2u/dx2")
            self.add_item(1, -nu / rho, "d2u/dy2")
            self.add_item(1, 1.0 / rho, "dw/dx")
            # momentum y
            self.add_item(2, 1.0, "u", "dv/dx")
            self.add_item(2, 1.0, "v", "dv/dy")
            self.add_item(2, -nu / rho, "d2v/dx2")
            self.add_item(2, -nu / rho, "d2v/dy2")
            self.add_item(2, 1.0 / rho, "dw/dy")
        elif dim == 2 and time_dependent == True:
            # continuty 
            self.add_item(0, 1.0, "du/dx")
            self.add_item(0, 1.0, "dv/dy")
            # momentum x
            self.add_item(1, 1.0, "du/dt")
            self.add_item(1, 1.0, "u", "du/dx")
            self.add_item(1, 1.0, "v", "du/dy")
            self.add_item(1, -nu / rho, "d2u/dx2")
            self.add_item(1, -nu / rho, "d2u/dy2")
            self.add_item(1, 1.0 / rho, "dw/dx")
            # momentum y
            self.add_item(2, 1.0, "dv/dt")
            self.add_item(2, 1.0, "u", "dv/dx")
            self.add_item(2, 1.0, "v", "dv/dy")
            self.add_item(2, -nu / rho, "d2v/dx2")
            self.add_item(2, -nu / rho, "d2v/dy2")
            self.add_item(2, 1.0 / rho, "dw/dy")
        elif dim == 3 and time_dependent == False:
            # continuty 
            self.add_item(0, 1.0, "du/dx")
            self.add_item(0, 1.0, "dv/dy")
            self.add_item(0, 1.0, "dw/dz")
            # momentum x
            self.add_item(1, 1.0, "u", "du/dx")
            self.add_item(1, 1.0, "v", "du/dy")
            self.add_item(1, 1.0, "w", "du/dz")
            self.add_item(1, -nu / rho, "d2u/dx2")
            self.add_item(1, -nu / rho, "d2u/dy2")
            self.add_item(1, -nu / rho, "d2u/dz2")
            self.add_item(1, 1.0 / rho, "dp/dx")
            # momentum y
            self.add_item(2, 1.0, "u", "dv/dx")
            self.add_item(2, 1.0, "v", "dv/dy")
            self.add_item(2, 1.0, "w", "dv/dz")
            self.add_item(2, -nu / rho, "d2v/dx2")
            self.add_item(2, -nu / rho, "d2v/dy2")
            self.add_item(2, -nu / rho, "d2v/dz2")
            self.add_item(2, 1.0 / rho, "dp/dy")
            # momentum z
            self.add_item(3, 1.0, "u", "dw/dx")
            self.add_item(3, 1.0, "v", "dw/dy")
            self.add_item(3, 1.0, "w", "dw/dz")
            self.add_item(3, -nu / rho, "d2w/dx2")
            self.add_item(3, -nu / rho, "d2w/dy2")
            self.add_item(3, -nu / rho, "d2w/dz2")
            self.add_item(3, 1.0 / rho, "dp/dz")
        elif dim == 3 and time_dependent == True:
            # continuty 
            self.add_item(0, 1.0, "du/dx")
            self.add_item(0, 1.0, "dv/dy")
            self.add_item(0, 1.0, "dw/dz")
            # momentum x
            self.add_item(1, 1.0, "du/dt")
            self.add_item(1, 1.0, "u", "du/dx")
            self.add_item(1, 1.0, "v", "du/dy")
            self.add_item(1, 1.0, "w", "du/dz")
            self.add_item(1, -nu / rho, "d2u/dx2")
            self.add_item(1, -nu / rho, "d2u/dy2")
            self.add_item(1, -nu / rho, "d2u/dz2")
            self.add_item(1, 1.0 / rho, "dp/dx")
            # momentum y
            self.add_item(2, 1.0, "dv/dt")
            self.add_item(2, 1.0, "u", "dv/dx")
            self.add_item(2, 1.0, "v", "dv/dy")
            self.add_item(2, 1.0, "w", "dv/dz")
            self.add_item(2, -nu / rho, "d2v/dx2")
            self.add_item(2, -nu / rho, "d2v/dy2")
            self.add_item(2, -nu / rho, "d2v/dz2")
            self.add_item(2, 1.0 / rho, "dp/dy")
            # momentum z
            self.add_item(3, 1.0, "dw/dt")
            self.add_item(3, 1.0, "u", "dw/dx")
            self.add_item(3, 1.0, "v", "dw/dy")
            self.add_item(3, 1.0, "w", "dw/dz")
            self.add_item(3, -nu / rho, "d2w/dx2")
            self.add_item(3, -nu / rho, "d2w/dy2")
            self.add_item(3, -nu / rho, "d2w/dz2")
            self.add_item(3, 1.0 / rho, "dp/dz")

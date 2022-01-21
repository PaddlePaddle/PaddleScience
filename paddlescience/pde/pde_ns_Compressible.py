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


class NavierStokes_Compressed(PDE):
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

    def __init__(self):
        dim = 2
        gamma = 1.4
        super(NavierStokes_Compressed, self).__init__(4)
        if dim == 2:

            # continuty
            self.add_item(0, 1.0, "w", "du/dx")
            self.add_item(0, 1.0, "u", "dw/dx")
            self.add_item(0, 1.0, "w", "dv/dy")
            self.add_item(0, 1.0, "v", "dw/dy")

            # momentum x
            self.add_item(1, 2.0, "w", "du/dx")
            self.add_item(1, 1.0, "u", "u","dw/dx")
            self.add_item(1, 1.0, "dp/dx")
            self.add_item(1, 1.0, "w", "u","dv/dy")
            self.add_item(1, 1.0, "w", "v", "du/dy")
            self.add_item(1, 1.0, "u", "v","dw/dy")

            # momentum y
            self.add_item(2, 1.0, "w", "u", "dv/dx")
            self.add_item(2, 1.0, "w", "v", "du/dx")
            self.add_item(2, 1.0, "u", "v", "dw/dx")
            self.add_item(2, 2.0, "w", "dv/dy")
            self.add_item(2, 1.0, "v", "v", "dw/dy")
            self.add_item(2, 1.0, "dp/dy")
            # energy
            self.add_item(3, gamma / (gamma - 1), "p", "du/dx")
            self.add_item(3, gamma / (gamma - 1), "u", "dp/dx")
            self.add_item(3, 1.0, "w", "u", "u", "du/dx")
            self.add_item(3, 1.0, "w", "u", "v", "dv/dx")
            self.add_item(3, 0.5, "u", "u", "w", "du/dx")
            self.add_item(3, 0.5, "u", "u", "u", "dw/dx")
            self.add_item(3, 0.5, "v", "v", "w", "du/dx")
            self.add_item(3, 0.5, "v", "v", "u", "dw/dx")

            self.add_item(3, gamma / (gamma - 1), "p", "dv/dy")
            self.add_item(3, gamma / (gamma - 1), "v", "dp/dy")
            self.add_item(3, 1.0, "w", "v", "u", "du/dy")
            self.add_item(3, 1.0, "w", "v", "v", "dv/dy")
            self.add_item(3, 0.5, "u", "u", "w", "dv/dy")
            self.add_item(3, 0.5, "u", "u", "v", "dw/dy")
            self.add_item(3, 0.5, "v", "v", "w", "dv/dy")
            self.add_item(3, 0.5, "v", "v", "v", "dw/dy")


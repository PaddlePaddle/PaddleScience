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

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class AdvectionDiffusion(base.PDE):
    r"""Class for Advection-Diffusion equation.

    Args:
        temperature (str): The temperature variable name used in the equation. Defaults to "T".
        diffusivity (str): The diffusivity coefficient name used in the equation. Defaults to "D".
        source_term (float): The source term of the equation. Defaults to 0.
        rho (str): The density variable name used in the equation. Defaults to "rho".
        dim (int): The dimension of the equation. Defaults to 3.
        time (bool): Whether or not to include time as a variable. Defaults to False.
        mixed_form (bool): Whether or not to use mixed partial derivatives in the diffusion term. Defaults to False.
        couple_method (str): The method used to couple 'heat_only', 'momentum_only', or 'heat_and_momentum'. Defaults to 'heat_only'.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.(temperature="c", diffusivity=diffusivity,rho=1.0, dim=2, time=False)
    """
    def __init__(
        self, temperature="c", diffusivity="D", source_term=0, rho="rho", dim=3, time=False,
        mixed_form=False, couple_method='heat_only'
    ):
        super().__init__()
        # set params
        self.T = temperature
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        self.diffusivity = diffusivity
        self.source_term = source_term

        def advection_diffusion_func(out):
            x, y = out["x"], out["y"]
            if couple_method == 'heat_only':
                out["u"], out["v"] = out["u"].detach(), out["v"].detach()
            u, v = out["u"], out["v"]
            T = out[self.T]
            D = self.diffusivity
            Q = self.source_term

            advection = (
                rho * u * jacobian(T, x) + 
                rho * v * jacobian(T, y)
            )

            if not self.mixed_form:
                diffusion = (
                (rho * D * hessian(T, x)) +
                (rho * D * hessian(T, y))
                )
            return advection - diffusion - Q
        self.add_equation("advection_diffusion_" + self.T, advection_diffusion_func)

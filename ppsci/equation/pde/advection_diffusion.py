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
    def __init__(
        self, temperature="T", diffusivity="D", source_term=0, rho="rho", dim=3, time=False, mixed_form=False
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
            u, v = out["u"], out["v"]
            T = out["c"]
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

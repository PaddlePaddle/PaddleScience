# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

from typing import Optional
from typing import Tuple

from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class AllenCahn(base.PDE):
    r"""Class for Allen-Cahn equation.

    $$
    \dfrac{\partial u}{\partial t} - \epsilon^2 \Delta u + 5u^3 - 5u = 0
    $$

    Args:
        eps (float): Represents the characteristicscale of interfacial width,
            influencing the thickness and dynamics of phase boundaries.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.AllenCahn(0.01**2)
    """

    def __init__(
        self,
        eps: float,
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys
        self.eps = eps
        # t, x = self.create_symbols("t x")
        # invars = (t, x, )
        # u = self.create_function("u", invars)

        # allen_cahn = u.diff(t) + 5 * u**3 - 5 * u - 0.0001 * u.diff(x, 2)

        # TODO: Pow(u,3) seems cause slightly larger L2 error than multiply(u*u*u)
        def allen_cahn(out):
            t, x = out["t"], out["x"]
            u = out["u"]
            u__t, u__x = jacobian(u, [t, x])
            u__x__x = jacobian(u__x, x)

            return u__t - self.eps * u__x__x + 5 * u * u * u - 5 * u

        self.add_equation("allen_cahn", allen_cahn)

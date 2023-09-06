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

from __future__ import annotations

from ppsci.equation.pde import base


class Poisson(base.PDE):
    r"""Class for poisson equation.

    $$
    \nabla^2 \varphi = C
    $$

    Args:
        dim (int): Dimension of equation.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Poisson(2)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        invars = self.create_symbols("x y z")[: self.dim]
        p = self.create_function("p", invars)

        poisson = 0
        for invar in invars:
            poisson += p.diff(invar).diff(invar)

        self.add_equation("poisson", poisson)

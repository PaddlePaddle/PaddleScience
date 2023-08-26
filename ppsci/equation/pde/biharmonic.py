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

from ppsci.equation.pde import base


class Biharmonic(base.PDE):
    r"""Class for biharmonic equation.

    $$
    \nabla^4 \varphi = \dfrac{q}{D}
    $$

    Args:
        dim (int): Dimension of equation.
        q (float): Load.
        D (float): Rigidity.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Biharmonic(2, -1.0, 1.0)
    """

    def __init__(self, dim: int, q: float, D: float):
        super().__init__()
        self.dim = dim
        self.q = q
        self.D = D

        invars = self.create_symbols(("x", "y", "z")[: self.dim])
        u = self.create_function("u", invars)
        biharmonic = -self.q / self.D
        for invar_i in invars:
            for invar_j in invars:
                biharmonic += u.diff(invar_i).diff(invar_i).diff(invar_j).diff(invar_j)

        self.add_equation("biharmonic", biharmonic)

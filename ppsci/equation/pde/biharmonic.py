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

from typing import Optional
from typing import Tuple
from typing import Union

import sympy

from ppsci.equation.pde import base


class Biharmonic(base.PDE):
    r"""Class for biharmonic equation with supporting special load.

    $$
    \nabla^4 \varphi = \dfrac{q}{D}
    $$

    Args:
        dim (int): Dimension of equation.
        q (Union[float, str, sympy.Basic]): Load.
        D (Union[float, str]): Rigidity.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Biharmonic(2, -1.0, 1.0)
    """

    def __init__(
        self,
        dim: int,
        q: Union[float, str, sympy.Basic],
        D: Union[float, str],
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.detach_keys = detach_keys

        invars = self.create_symbols("x y z")[:dim]
        u = self.create_function("u", invars)

        if isinstance(q, str):
            q = self.create_function("q", invars)
        if isinstance(D, str):
            D = self.create_function("D", invars)

        self.dim = dim
        self.q = q
        self.D = D

        biharmonic = -self.q / self.D
        for invar_i in invars:
            for invar_j in invars:
                biharmonic += u.diff(invar_i, 2).diff(invar_j, 2)

        self.add_equation("biharmonic", biharmonic)

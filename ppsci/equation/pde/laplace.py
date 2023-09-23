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

from ppsci.equation.pde import base


class Laplace(base.PDE):
    r"""Class for laplace equation.

    $$
    \nabla^2 \varphi = 0
    $$

    Args:
        dim (int): Dimension of equation.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Laplace(2)
    """

    def __init__(self, dim: int, detach_keys: Optional[Tuple[str, ...]] = None):
        super().__init__()
        self.detach_keys = detach_keys

        invars = self.create_symbols("x y z")[:dim]
        u = self.create_function("u", invars)

        self.dim = dim

        laplace = 0
        for invar in invars:
            laplace += u.diff(invar, 2)

        self.add_equation("laplace", laplace)

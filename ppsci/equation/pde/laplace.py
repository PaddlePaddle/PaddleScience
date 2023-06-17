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
from ppsci.equation.pde import base


class Laplace(base.PDE):
    """Class for laplace equation.

    Args:
        dim (int): Dimension of equation.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Laplace(2)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        def laplace_compute_func(out):
            x, y = out["x"], out["y"]
            u = out["u"]
            laplace = hessian(u, x) + hessian(u, y)
            if self.dim == 3:
                z = out["z"]
                laplace += hessian(u, z)
            return laplace

        self.add_equation("laplace", laplace_compute_func)

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


class Biharmonic(base.PDE):
    """Biharmonic equation.

    Args:
        dim (int): Dimension of equation.
        q (float): Load.
        D (float): Rigidity.
    """

    def __init__(self, dim: int, q: float, D: float):
        super().__init__()
        self.dim = dim
        self.q = q
        self.D = D

        def biharmonic_compute_func(out):
            u = out["u"]

            keys = ["x", "y", "z"]
            biharmonic = -self.q / self.D
            for i in range(self.dim):
                key = keys[i]
                u__ = hessian(u, out[key])
                for j in range(self.dim):
                    key = keys[j]
                    biharmonic += hessian(u__, out[key])
            return biharmonic

        self.add_equation("biharmonic", biharmonic_compute_func)

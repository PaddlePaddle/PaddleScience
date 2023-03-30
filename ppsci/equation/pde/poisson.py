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


class Poisson(base.PDE):
    """Poisson

    Args:
        dim (int): Number of dimension.
        alpha (float): Alpha factor.
        time (bool): Whther equation is time-dependent.
    """

    def __init__(self, dim, alpha, time):
        super().__init__()
        t, x, y, z = self.create_symbols("t x y z")
        invars = [x, y, z][:dim]
        if time:
            invars = [t] + invars

        u = self.create_function("u", invars)

        self.equations["poisson"] = u.diff(t) - alpha * (
            u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        )

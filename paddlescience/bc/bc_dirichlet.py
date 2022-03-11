# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class BC:
    def __init__(self, name):
        self.category = None
        self.name = name


class Dirichlet(BC):
    def __init__(self, name):
        super(Dirichlet, self).__init__(name)
        self.category = "Dirichlet"

    def compute(self, u, du=None, dn=None, value=None):
        return paddle.norm(u - value, p=2)


class Neumann(BC):
    def __init__(self, name):
        super(Neumann, self).__init__(name)
        self.category = "Neumann"

    # dn: normal direction
    def compute(self, u, du=None, dn=None, value=None):
        return paddle.norm(du * dn - value, p=2)


class Robin(BC):
    def __init__(self, name):
        super(Robin, self).__init__(name)
        self.category = "Robin"

    def compute(self, u, du=None, dn=None, value=None):
        diff = u + du * dn - value
        return paddle.norm(diff, p=2)


if __name__ == "__main__":

    pde = psci.pde.NavierStokes()

    bc1 = psci.bc.Dirichlet(name="bc1")
    bc2 = psci.bc.Dirichlet(name="bc2")
    bc3 = psci.bc.Neumann(name="bc3")

    pde.add_bc(bc1, bc2, bc3)

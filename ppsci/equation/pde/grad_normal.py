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

class GradNormal(base.PDE):
    def __init__(self, T, dim=3, time=True):
        super().__init__()
        self.T = T
        self.dim = dim
        self.time = time

        def normal_gradient_fun(out):
            x, y = out["x"], out["y"]
            T = out[self.T]
            normal_x, normal_y = out["normal_x"], out["normal_x"]
            return normal_x * jacobian(T, x) + normal_y * jacobian(T, y)
        
        self.add_equation("normal_gradient_" + self.T, normal_gradient_fun)

        


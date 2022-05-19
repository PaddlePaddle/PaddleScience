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

import paddlescience as psci
import sympy

# PDE
x = sympy.Symbol('x')
y = sympy.Symbol('y')
u = sympy.Function('u')(x, y)
pde = psci.pde.PDE(num_equations=1, time_dependent=False, order=2)
pde.indvar = [x, y]
pde.dvar = [u]
pde.equations[0] = u.diff(x).diff(x) + u.diff(y).diff(y)
pde.rhs[0] = 0.0

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

from ppsci.equation.pde.base import PDE
from ppsci.equation.pde.biharmonic import Biharmonic
from ppsci.equation.pde.laplace import Laplace
from ppsci.equation.pde.navier_stokes import NavierStokes
from ppsci.equation.pde.normal_dot_vec import NormalDotVec
from ppsci.equation.pde.poisson import Poisson
from ppsci.equation.pde.viv import Vibration

__all__ = [
    "PDE",
    "Biharmonic",
    "Laplace",
    "NavierStokes",
    "NormalDotVec",
    "Poisson",
    "Vibration",
]

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

from typing import Tuple

from ppsci.equation.pde import base


class NormalDotVec(base.PDE):
    """Poisson

    Args:
        velocity_keys (Tuple[str, ...]): Keys for velocity(ies).
    """

    def __init__(self, velocity_keys: Tuple[str, ...]):
        super().__init__()
        normal_x, normal_y, normal_z = self.create_symbols("normal_x normal_y normal_z")
        outvars = [self.create_symbols(v) for v in velocity_keys]
        normals = [normal_x, normal_y, normal_z]

        self.equations["normal_dot_vel"] = 0
        for i, velocity in enumerate(outvars):
            self.equations["normal_dot_vel"] += velocity * normals[i]

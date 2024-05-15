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


class NormalDotVec(base.PDE):
    r"""Normal Dot Vector.

    $$
    \mathbf{n} \cdot \mathbf{v} = 0
    $$

    Args:
        vec_keys (Tuple[str, ...]): Keys for vectors, such as ("u", "v", "w") for
            velocity vector.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.NormalDotVec(("u", "v", "w"))
    """

    def __init__(
        self, vec_keys: Tuple[str, ...], detach_keys: Optional[Tuple[str, ...]] = None
    ):
        super().__init__()
        self.detach_keys = detach_keys
        if not vec_keys:
            raise ValueError(f"len(vec_keys)({len(vec_keys)}) should be larger than 0.")

        self.vec_keys = vec_keys
        vec_vars = self.create_symbols(" ".join(vec_keys))
        normals = self.create_symbols("normal_x normal_y normal_z")

        normal_dot_vec = 0
        for (normal, vec) in zip(normals, vec_vars):
            normal_dot_vec += normal * vec

        self.add_equation("normal_dot_vec", normal_dot_vec)

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

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import paddle.autograd

from ppsci.equation.pde import base


# forward over forward
def hvp_fwdfwd(f, primals):
    g = lambda x, y, z: paddle.incubate.autograd.jvp(f, (x, y, z))[1]
    _, tangents_out = paddle.incubate.autograd.jvp(g, primals)
    return tangents_out


class Helmholtz(base.PDE):
    r"""Class for biharmonic equation with supporting special load.

    $$
    \nabla^4 \varphi = \dfrac{q}{D}
    $$

    Args:
        dim (int): Dimension of equation.
        q (Union[float, str, sympy.Basic]): Load.
        D (Union[float, str]): Rigidity.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Biharmonic(2, -1.0, 1.0)
    """

    def __init__(
        self,
        dim: int,
        lda: float,
        source: Union[str, float],
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.lda = lda
        self.detach_keys = detach_keys

        # invars = self.create_symbols("x y z")[:dim]
        # u = self.create_function("u", invars)
        # if isinstance(source, str):
        # source = self.create_symbols(source)

        # helmholtz = self.lda * u
        # for var in invars:
        #     helmholtz += u.diff(var, 2)
        self.model: paddle.nn.Layer

        def helmholtz(data_dict: Dict[str, "paddle.Tensor"]):
            x, y, z = (
                data_dict["x"],
                data_dict["y"],
                data_dict["z"],
            )  # [n1, ], [n2, ], [n3, ]
            u = data_dict["u"]  # [n1n2n3, 1]

            u__x__x = hvp_fwdfwd(self.model.forward_tensor, (x, y, z))  # [n1n2n3, 1]
            u__y__y = hvp_fwdfwd(self.model.forward_tensor, (x, y, z))  # [n1n2n3, 1]
            u__z__z = hvp_fwdfwd(self.model.forward_tensor, (x, y, z))  # [n1n2n3, 1]

            out = self.lda * u + u__x__x + u__y__y + u__z__z
            return out

        self.add_equation("helmholtz", helmholtz)

        self._apply_detach()

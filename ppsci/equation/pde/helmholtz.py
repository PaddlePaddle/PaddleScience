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

from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import paddle

from ppsci.equation.pde import base


def hvp_revrev(f: Callable, primals: Tuple[paddle.Tensor, ...]) -> paddle.Tensor:
    """Compute the Hessian vector product of f with respect to primals using
        double backward trick in reverse mode AD.

    Args:
        f (Callable): Function to compute HVP.
        primals (Tuple[paddle.Tensor, ...]): Input tensors.

    Returns:
        paddle.Tensor: Hessian vector product of f with respect to primals.
    """
    # TODO: Merge this option into ppsci.autodiff.ad
    g = lambda primals: paddle.incubate.autograd.jvp(f, primals)[1]
    tangents_out = paddle.incubate.autograd.jvp(g, primals)[1]
    return tangents_out[0]


class Helmholtz(base.PDE):
    r"""Class for helmholtz equation.

    $$
    \nabla^2 u + k^2 u = f
    $$

    $$
    \text{where } f \text{ is the source term}.
    $$

    Args:
        dim (int): Dimension of equation.
        detach_keys (Optional[Tuple[str, ...]]): Keys used for detach during computing.
            Defaults to None.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.Helmholtz(2, -1.0, 1.0)
    """

    def __init__(
        self,
        dim: int,
        k: float,
        model: paddle.nn.Layer,
        detach_keys: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.detach_keys = detach_keys

        invars = self.create_symbols("x y z")[:dim]

        # TODO: This is a hack, should be simplified in the future
        self.model = model

        def helmholtz(data_dict: Dict[str, paddle.Tensor]) -> paddle.Tensor:
            xs = tuple(data_dict[invar.name] for invar in invars)

            # TODO: Hard code here, for hvp_revrev requires tuple input(s) but not dict
            if self.dim == 1:
                u__x__x = hvp_revrev(lambda x_: self.model.forward_tensor(x_), (xs[0],))
                out = (self.k**2) * data_dict["u"] + u__x__x
            elif self.dim == 2:
                u__x__x = hvp_revrev(
                    lambda x_: self.model.forward_tensor(x_, xs[1]), (xs[0],)
                )
                u__y__y = hvp_revrev(
                    lambda y_: self.model.forward_tensor(xs[0], y_), (xs[1],)
                )
                out = (self.k**2) * data_dict["u"] + u__x__x + u__y__y
            elif self.dim >= 3:
                u__x__x = hvp_revrev(
                    lambda x_: self.model.forward_tensor(x_, xs[1], xs[2]), (xs[0],)
                )
                u__y__y = hvp_revrev(
                    lambda y_: self.model.forward_tensor(xs[0], y_, xs[2]), (xs[1],)
                )
                u__z__z = hvp_revrev(
                    lambda z_: self.model.forward_tensor(xs[0], xs[1], z_), (xs[2],)
                )
                out = (self.k**2) * data_dict["u"] + u__x__x + u__y__y + u__z__z
            else:
                raise NotImplementedError(
                    f"dim should be less or equal to 3, but got {self.dim}."
                )

            return out

        self.add_equation("helmholtz", helmholtz)

        self._apply_detach()

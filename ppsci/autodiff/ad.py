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

"""
This module is adapted from [https://github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde)
"""

from typing import Optional

import paddle


class _Jacobian:
    """Compute Jacobian matrix J: J[i][j] = dy_i/dx_j, where i = 0, ..., dim_y-1 and
    j = 0, ..., dim_x - 1.

    It is lazy evaluation, i.e., it only computes J[i][j] when needed, and will cache
    by output tensor(row index in jacobian matrix).

    Args:
        ys (paddle.Tensor): Output Tensor of shape [batch_size, dim_y].
        xs (paddle.Tensor): Input Tensor of shape [batch_size, dim_x].
    """

    def __init__(self, ys: "paddle.Tensor", xs: "paddle.Tensor"):
        self.ys = ys
        self.xs = xs

        self.dim_y = ys.shape[1]
        self.dim_x = xs.shape[1]

        self.J = {}

    def __call__(self, i: int = 0, j: Optional[int] = None) -> "paddle.Tensor":
        """Returns J[`i`][`j`]. If `j` is ``None``, returns the gradient of y_i, i.e.,
        J[i].
        """
        if not 0 <= i < self.dim_y:
            raise ValueError(f"i({i}) should in range [0, {self.dim_y}).")
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError(f"j({j}) should in range [0, {self.dim_x}).")
        # Compute J[i]
        if i not in self.J:
            y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
            self.J[i] = paddle.grad(y, self.xs, create_graph=True)[0]

        return self.J[i] if (j is None or self.dim_x == 1) else self.J[i][:, j : j + 1]


class Jacobians:
    r"""Compute multiple Jacobians.

    $$
    \rm Jacobian(ys, xs, i, j) = \dfrac{\partial ys_i}{\partial xs_j}
    $$

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Js = {}

    def __call__(
        self,
        ys: "paddle.Tensor",
        xs: "paddle.Tensor",
        i: int = 0,
        j: Optional[int] = None,
    ) -> "paddle.Tensor":
        """Compute jacobians for given ys and xs.

        Args:
            ys (paddle.Tensor): Output tensor.
            xs (paddle.Tensor): Input tensor.
            i (int, optional): i-th output variable. Defaults to 0.
            j (Optional[int]): j-th input variable. Defaults to None.

        Returns:
            paddle.Tensor: Jacobian matrix of ys[i] to xs[j].

        Examples:
            >>> import paddle
            >>> import ppsci
            >>> x = paddle.randn([4, 1])
            >>> x.stop_gradient = False
            >>> y = x * x
            >>> dy_dx = ppsci.autodiff.jacobian(y, x)
        """
        key = (ys, xs)
        if key not in self.Js:
            self.Js[key] = _Jacobian(ys, xs)
        return self.Js[key](i, j)

    def _clear(self):
        """Clear cached Jacobians."""
        self.Js = {}


# Use high-order differentiation with singleton pattern for convenient
jacobian = Jacobians()


class _Hessian:
    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j = 0,..., dim_x-1.

    It is lazy evaluation, i.e., it only computes H[i][j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, 1) or (batch_size, dim_y > 1).
        xs: Input Tensor of shape (batch_size, dim_x).
        component: If `y` has the shape (batch_size, dim_y > 1), then `y[:, component]`
            is used to compute the Hessian. Do not use if `y` has the shape (batch_size,
            1).
        grad_y: The gradient of `y` w.r.t. `xs`. Provide `grad_y` if known to avoid
            duplicate computation. `grad_y` can be computed from ``Jacobian``.
    """

    def __init__(
        self,
        ys: "paddle.Tensor",
        xs: "paddle.Tensor",
        component: Optional[int] = None,
        grad_y: Optional["paddle.Tensor"] = None,
    ):
        dim_y = ys.shape[1]

        if dim_y > 1:
            if component is None:
                raise ValueError(
                    f"component({component}) can not be None when dim_y({dim_y})>1."
                )
            if component >= dim_y:
                raise ValueError(
                    f"component({component}) should be smaller than dim_y({dim_y})."
                )
        else:
            if component is not None:
                raise ValueError(
                    f"component{component} should be set to None when dim_y({dim_y})=1."
                )
            component = 0

        if grad_y is None:
            grad_y = jacobian(ys, xs, i=component, j=None)
        self.H = _Jacobian(grad_y, xs)

    def __call__(self, i: int = 0, j: int = 0):
        """Returns H[`i`][`j`]."""
        return self.H(i, j)


class Hessians:
    r"""Compute multiple Hessians.

    $$
    \rm Hessian(ys, xs, component, i, j) = \dfrac{\partial ys_{component}}{\partial xs_i \partial xs_j}
    $$

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """

    def __init__(self):
        self.Hs = {}

    def __call__(
        self,
        ys: "paddle.Tensor",
        xs: "paddle.Tensor",
        component: Optional[int] = None,
        i: int = 0,
        j: int = 0,
        grad_y: Optional["paddle.Tensor"] = None,
    ) -> "paddle.Tensor":
        """Compute hessian matrix for given ys and xs.

        Args:
            ys (paddle.Tensor): Output tensor.
            xs (paddle.Tensor): Input tensor.
            component (Optional[int]): If `y` has the shape (batch_size, dim_y > 1), then `y[:, component]`
                is used to compute the Hessian. Do not use if `y` has the shape (batch_size,
                1). Defaults to None.
            i (int, optional): i-th input variable. Defaults to 0.
            j (int, optional): j-th input variable. Defaults to 0.
            grad_y (Optional[paddle.Tensor]): The gradient of `y` w.r.t. `xs`. Provide `grad_y` if known to avoid
                duplicate computation. Defaults to None.

        Returns:
            paddle.Tensor: Hessian matrix.

        Examples:
            >>> import paddle
            >>> import ppsci
            >>> x = paddle.randn([4, 3])
            >>> x.stop_gradient = False
            >>> y = (x * x).sin()
            >>> dy_dxx = ppsci.autodiff.hessian(y, x, component=0)
        """
        key = (ys, xs, component)
        if key not in self.Hs:
            self.Hs[key] = _Hessian(ys, xs, component=component, grad_y=grad_y)
        return self.Hs[key](i, j)

    def _clear(self):
        """Clear cached Hessians."""
        self.Hs = {}


# Use high-order differentiation with singleton pattern for convenient
hessian = Hessians()


def clear():
    """Clear cached Jacobians and Hessians."""
    jacobian._clear()
    hessian._clear()

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

import numpy as np
import paddle

from ppsci.equation.pde import PDE


class Volterra(PDE):
    r"""A second kind of volterra integral equation with Gaussian quadrature algorithm.

    $$
    x(t) - f(t)=\int_a^t K(t, s) x(s) d s
    $$

    [Volterra integral equation](https://en.wikipedia.org/wiki/Volterra_integral_equation)

    [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval)

    Args:
        bound (float): Lower bound `a` for Volterra integral equation.
        num_points (int): Sampled points in integral interval.
        quad_deg (int): Number of quadrature.
        kernel_func (Callable): Kernel func `K(t,s)`.
        func (Callable): `x(t) - f(t)` in Volterra integral equation.

    Examples:
        >>> import ppsci
        >>> import numpy as np
        >>> vol_eq = ppsci.equation.Volterra(
        ...     0, 12, 20, lambda t, s: np.exp(s - t), lambda out: out["u"],
        ... )
    """

    dtype = paddle.get_default_dtype()

    def __init__(
        self,
        bound: float,
        num_points: int,
        quad_deg: int,
        kernel_func: Callable,
        func: Callable,
    ):
        super().__init__()
        self.bound = bound
        self.num_points = num_points
        self.quad_deg = quad_deg
        self.kernel_func = kernel_func
        self.func = func

        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)
        self.quad_x = self.quad_x.astype(Volterra.dtype).reshape([-1, 1])  # [Q, 1]
        self.quad_x = paddle.to_tensor(self.quad_x)  # [Q, 1]

        self.quad_w = self.quad_w.astype(Volterra.dtype)  # [Q, ]

        def compute_volterra_func(out):
            x, u = out["x"], out["u"]
            lhs = self.func(out)

            int_mat = paddle.to_tensor(self._get_int_matrix(x), stop_gradient=False)
            rhs = paddle.mm(int_mat, u)  # (N, 1)

            volterra = lhs[: len(rhs)] - rhs
            return volterra

        self.add_equation("volterra", compute_volterra_func)

    def get_quad_points(self, t: paddle.Tensor) -> paddle.Tensor:
        """Scale and transform quad_x from [-1, 1] to range [a, b].
        reference: https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval

        Args:
            t (paddle.Tensor): Tensor array of upper bounds 't' for integral.

        Returns:
            paddle.Tensor: Transformed points in desired range with shape of [N, Q].
        """
        a, b = self.bound, t
        return ((b - a) / 2) @ self.quad_x.T + (b + a) / 2

    def _get_quad_weights(self, t: float) -> np.ndarray:
        """Scale weights to range according to given t and lower bound of integral.
        reference: https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval

        Args:
            t (float): Array of upper bound 't' for integral.

        Returns:
            np.ndarray: Transformed weights in desired range with shape of [Q, ].
        """
        a, b = self.bound, t
        return (b - a) / 2 * self.quad_w

    def _get_int_matrix(self, x: np.ndarray) -> np.ndarray:
        int_mat = np.zeros(
            (self.num_points, self.num_points + (self.num_points * self.quad_deg)),
            dtype=Volterra.dtype,
        )
        for i in range(self.num_points):
            xi = float(x[i])
            beg = self.num_points + self.quad_deg * i
            end = self.num_points + self.quad_deg * (i + 1)
            K = np.ravel(
                self.kernel_func(np.full((self.quad_deg, 1), xi), x[beg:end].numpy())
            )
            int_mat[i, beg:end] = self._get_quad_weights(xi) * K
        return int_mat

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

from typing import Callable

import numpy as np
import paddle

from ppsci.autodiff import jacobian
from ppsci.equation.pde import PDE


class Volterra(PDE):
    """Base class for Partial Differential Equation"""

    def __init__(self, num_points: int, quad_deg: int, kernel_func: Callable):
        super().__init__()
        self.num_points = num_points
        self.quad_deg = quad_deg
        self.kernel_func = kernel_func
        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)

        self.quad_x = self.quad_x.astype(paddle.get_default_dtype()).reshape([-1, 1])
        self.quad_w = self.quad_w.astype(paddle.get_default_dtype())
        self.quad_x = paddle.to_tensor(self.quad_x)

        def compute_volterra_func(out):
            x, u = out["x"], out["u"]
            int_mat = paddle.to_tensor(self.get_int_matrix(x), stop_gradient=False)
            rhs = paddle.mm(int_mat, u)  # (nide, 1)
            du_dx = jacobian(u, x)
            volterra = (du_dx + u)[: len(rhs)] - rhs
            return volterra

        self.add_equation("volterra", compute_volterra_func)

    def get_quad_points(self, x: float) -> np.ndarray:
        """Transform points from [N, 1] to desired range.

        Args:
            x (float): Points of shape [N, 1].

        Returns:
            np.ndarray: Transformed points in desired range with shape of [N, M].
        """
        return x @ (self.quad_x.T + 1) / 2

    def get_quad_weights(self, x: float) -> np.ndarray:
        """Transform weights to desired range.

        Args:
            x (float): [N, 1]

        Returns:
            np.ndarray: Output weights of shape [N, M].
        """
        return self.quad_w * x / 2

    def get_int_matrix(self, x: np.ndarray) -> np.ndarray:
        int_mat = np.zeros(
            (self.num_points, self.num_points + (self.num_points * self.quad_deg)),
            dtype=paddle.get_default_dtype(),
        )
        for i in range(self.num_points):
            xi = float(x[i, 0])
            beg = self.num_points + self.quad_deg * i
            end = self.num_points + self.quad_deg * (i + 1)
            K = np.ravel(
                self.kernel_func(np.full((self.quad_deg, 1), xi), x[beg:end].numpy())
            )
            int_mat[i, beg:end] = self.get_quad_weights(xi) * K
        return int_mat

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

import math
from typing import Tuple

import numpy as np
import paddle
from paddle import sparse
from scipy import special

from ppsci import geometry
from ppsci.equation.pde import PDE
from ppsci.utils import misc


class FractionalPoisson(PDE):
    r"""

    TODO: refine this docstring
    Args:
        alpha (float): Alpha.
        geom (geometry.Geometry): Computation geometry.
        resolution (Tuple[int, ...]): Resolution.

    Examples:
        >>> import ppsci
        >>> geom_disk = ppsci.geometry.Disk([0, 0], 1)
        >>> ALPHA = 0.5
        >>> fpde = ppsci.equation.FractionalPoisson(ALPHA, geom_disk, [8, 100])
    """
    dtype = paddle.get_default_dtype()

    def __init__(
        self, alpha: float, geom: geometry.Geometry, resolution: Tuple[int, ...]
    ):
        super().__init__()
        self.alpha = alpha
        self.geom = geom
        self.resolution = resolution
        self._w_init = self._init_weights()

        def compute_fpde_func(out):
            x = paddle.concat((out["x"], out["y"]), axis=1)
            y = out["u"]
            indices, values, shape = self.int_mat
            int_mat = sparse.sparse_coo_tensor(
                [[p[0] for p in indices], [p[1] for p in indices]],
                values,
                shape,
                stop_gradient=False,
            )
            lhs = sparse.matmul(int_mat, y)
            lhs = lhs[:, 0]
            lhs *= (
                special.gamma((1 - self.alpha) / 2)
                * special.gamma((2 + self.alpha) / 2)
                / (2 * np.pi**1.5)
            )
            x = x[: paddle.numel(lhs)]
            rhs = (
                2**self.alpha
                * special.gamma(2 + self.alpha / 2)
                * special.gamma(1 + self.alpha / 2)
                * (1 - (1 + self.alpha / 2) * paddle.sum(x**2, axis=1))
            )
            res = lhs - rhs
            return res

        self.add_equation("fpde", compute_fpde_func)

    def _init_weights(self):
        n = self._dynamic_dist2npts(self.geom.diam) + 1
        w = [1.0]
        for j in range(1, n):
            w.append(w[-1] * (j - 1 - self.alpha) / j)
        return np.array(w, dtype=self.dtype)

    def get_x(self, x_f):
        if hasattr(self, "train_x"):
            return self.train_x

        self.x0 = x_f
        if np.any(self.geom.on_boundary(self.x0)):
            raise ValueError("x0 contains boundary points.")

        if self.geom.ndim == 1:
            dirns, dirn_w = [-1, 1], [1, 1]
        elif self.geom.ndim == 2:
            gauss_x, gauss_w = np.polynomial.legendre.leggauss(self.resolution[0])
            gauss_x, gauss_w = gauss_x.astype(self.dtype), gauss_w.astype(self.dtype)
            thetas = np.pi * gauss_x + np.pi
            dirns = np.vstack((np.cos(thetas), np.sin(thetas))).T
            dirn_w = np.pi * gauss_w
        elif self.geom.ndim == 3:
            gauss_x, gauss_w = np.polynomial.legendre.leggauss(max(self.resolution[:2]))
            gauss_x, gauss_w = gauss_x.astype(self.dtype), gauss_w.astype(self.dtype)
            thetas = (np.pi * gauss_x[: self.resolution[0]] + np.pi) / 2
            phis = np.pi * gauss_x[: self.resolution[1]] + np.pi
            dirns, dirn_w = [], []
            for i in range(self.resolution[0]):
                for j in range(self.resolution[1]):
                    dirns.append(
                        [
                            np.sin(thetas[i]) * np.cos(phis[j]),
                            np.sin(thetas[i]) * np.sin(phis[j]),
                            np.cos(thetas[i]),
                        ]
                    )
                    dirn_w.append(gauss_w[i] * gauss_w[j] * np.sin(thetas[i]))
            dirn_w = np.pi**2 / 2 * np.array(dirn_w)

        x, self.w = [], []
        for x0i in self.x0:
            xi = list(
                map(
                    lambda dirn: self.background_points(
                        x0i, dirn, self._dynamic_dist2npts, 0
                    ),
                    dirns,
                )
            )
            wi = list(
                map(
                    lambda i: dirn_w[i]
                    * np.linalg.norm(xi[i][1] - xi[i][0]) ** (-self.alpha)
                    * self.get_weight(len(xi[i]) - 1),
                    range(len(dirns)),
                )
            )
            # first order
            # xi, wi = zip(self.modify_first_order(xij, wij) for xij, wij in zip(xi, wi))
            xi, wi = zip(*map(self.modify_first_order, xi, wi))
            # second order
            # xi, wi = zip(*map(self.modify_second_order, xi, wi))
            # third order
            # xi, wi = zip(*map(self.modify_third_order, xi, wi))
            x.append(np.vstack(xi))
            self.w.append(np.hstack(wi))
        self.x = np.vstack([self.x0] + x)
        self.int_mat = self._get_int_matrix(self.x0)
        self.train_x = misc.convert_to_dict(self.x, ("x", "y"))
        return self.train_x

    def get_weight(self, n):
        return self._w_init[: n + 1]

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n - shift + 1, dtype=self.dtype)[:, None] * h * dirn
        return pts

    def distance2boundary_unitdirn(self, x, dirn):
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        xc = x - self.geom.center
        xc = xc
        ad = np.dot(xc, dirn)
        return (
            -ad + (ad**2 - np.sum(xc * xc, axis=-1) + self.geom.radius**2) ** 0.5
        ).astype(self.dtype)

    def modify_first_order(self, x, w):
        x = np.vstack(([2 * x[0] - x[1]], x[:-1]))
        if not self.geom.is_inside(x[0:1])[0]:
            return x[1:], w[1:]
        return x, w

    def _dynamic_dist2npts(self, dx):
        return int(math.ceil(self.resolution[-1] * dx))

    def _get_int_matrix(self, x: np.ndarray) -> np.ndarray:
        dense_shape = (x.shape[0], self.x.shape[0])
        indices, values = [], []
        beg = x.shape[0]
        for i in range(x.shape[0]):
            for _ in range(self.w[i].shape[0]):
                indices.append([i, beg])
                beg += 1
            values = np.hstack((values, self.w[i]))
        return indices, values.astype(self.dtype), dense_shape

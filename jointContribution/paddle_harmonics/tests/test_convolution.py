# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import math
import unittest
from functools import partial

import numpy as np
import paddle
from parameterized import parameterized

from paddle_harmonics import DiscreteContinuousConvS2
from paddle_harmonics import DiscreteContinuousConvTransposeS2
from paddle_harmonics import quadrature
from paddle_harmonics.utils import paddle_aux  # noqa, for reshape


def _compute_vals_isotropic(
    theta: paddle.Tensor, phi: paddle.Tensor, ntheta: int, theta_cutoff: float
):
    """
    helper routine to compute the values of the isotropic kernel densely
    """

    # compute the support
    dtheta = (theta_cutoff - 0.0) / ntheta
    ikernel = paddle.arange(ntheta).reshape(-1, 1, 1)
    itheta = ikernel * dtheta

    norm_factor = (
        2
        * math.pi
        * (
            1
            - math.cos(theta_cutoff - dtheta)
            + math.cos(theta_cutoff - dtheta)
            + (math.sin(theta_cutoff - dtheta) - math.sin(theta_cutoff)) / dtheta
        )
    )

    vals = paddle.where(
        ((theta - itheta).abs() <= dtheta) & (theta <= theta_cutoff),
        (1 - (theta - itheta).abs() / dtheta) / norm_factor,
        0.0,
    )
    return vals


def _compute_vals_anisotropic(
    theta: paddle.Tensor, phi: paddle.Tensor, ntheta: int, nphi: int, theta_cutoff: float
):
    """
    helper routine to compute the values of the anisotropic kernel densely
    """

    # compute the support
    dtheta = (theta_cutoff - 0.0) / ntheta
    dphi = 2.0 * math.pi / nphi
    kernel_size = (ntheta - 1) * nphi + 1
    ikernel = paddle.arange(kernel_size).reshape(-1, 1, 1)
    itheta = ((ikernel - 1) // nphi + 1) * dtheta
    iphi = ((ikernel - 1) % nphi) * dphi

    norm_factor = (
        2
        * math.pi
        * (
            1
            - math.cos(theta_cutoff - dtheta)
            + math.cos(theta_cutoff - dtheta)
            + (math.sin(theta_cutoff - dtheta) - math.sin(theta_cutoff)) / dtheta
        )
    )

    # find the indices where the rotated position falls into the support of the kernel
    cond_theta = ((theta - itheta).abs() <= dtheta) & (theta <= theta_cutoff)
    cond_phi = ((phi - iphi).abs() <= dphi) | ((2 * math.pi - (phi - iphi).abs()) <= dphi)
    theta_vals = paddle.where(cond_theta, (1 - (theta - itheta).abs() / dtheta) / norm_factor, 0.0)
    phi_vals = paddle.where(
        cond_phi,
        (1 - paddle.minimum((phi - iphi).abs(), (2 * math.pi - (phi - iphi).abs())) / dphi),
        0.0,
    )
    vals = paddle.where(ikernel > 0, theta_vals * phi_vals, theta_vals)
    return vals


def _precompute_convolution_tensor_dense(
    in_shape,
    out_shape,
    kernel_shape,
    grid_in="equiangular",
    grid_out="equiangular",
    theta_cutoff=0.01 * math.pi,
):
    """
    Helper routine to compute the convolution Tensor in a dense fashion
    """

    assert len(in_shape) == 2
    assert len(out_shape) == 2

    if len(kernel_shape) == 1:
        kernel_handle = partial(
            _compute_vals_isotropic, ntheta=kernel_shape[0], theta_cutoff=theta_cutoff
        )
        kernel_size = kernel_shape[0]
    elif len(kernel_shape) == 2:
        kernel_handle = partial(
            _compute_vals_anisotropic,
            ntheta=kernel_shape[0],
            nphi=kernel_shape[1],
            theta_cutoff=theta_cutoff,
        )
        kernel_size = (kernel_shape[0] - 1) * kernel_shape[1] + 1
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, _ = quadrature._precompute_latitudes(nlat_in, grid=grid_in)  # noqa
    lats_in = paddle.to_tensor(lats_in).astype("float32")
    lats_out, _ = quadrature._precompute_latitudes(nlat_out, grid=grid_out)  # noqa
    lats_out = paddle.to_tensor(lats_out).astype(
        "float32"
    )  # array for accumulating non-zero indices

    # compute the phi differences. We need to make the linspace exclusive to not double the last point
    lons_in = paddle.linspace(0, 2 * math.pi, nlon_in + 1)[:-1]
    lons_out = paddle.linspace(0, 2 * math.pi, nlon_out + 1)[:-1]

    out = paddle.zeros(shape=[kernel_size, nlat_out, nlon_out, nlat_in, nlon_in])

    for t in range(nlat_out):
        for p in range(nlon_out):
            alpha = -lats_out[t]
            beta = lons_in - lons_out[p]
            gamma = lats_in.reshape(-1, 1)

            # compute latitude of the rotated position
            z = -paddle.cos(beta) * paddle.sin(alpha) * paddle.sin(gamma) + paddle.cos(
                alpha
            ) * paddle.cos(gamma)

            # compute cartesian coordinates of the rotated position
            x = paddle.cos(alpha) * paddle.cos(beta) * paddle.sin(gamma) + paddle.cos(
                gamma
            ) * paddle.sin(alpha)
            y = paddle.sin(beta) * paddle.sin(gamma)

            # normalize instead of clipping to ensure correct range
            norm = paddle.sqrt(x * x + y * y + z * z)
            x = x / norm
            y = y / norm
            z = z / norm

            # compute spherical coordinates
            theta = paddle.acos(z)
            phi = paddle.atan2(y, x) + np.pi

            # find the indices where the rotated position falls into the support of the kernel
            out[:, t, p, :, :] = kernel_handle(theta, phi)

    return out


class TestDiscreteContinuousConvolution(unittest.TestCase):
    def setUp(self):
        if paddle.device.cuda.device_count() >= 1:
            self.device = paddle.CUDAPlace(0)
        else:
            self.device = paddle.CPUPlace()
        # self.device = paddle.CPUPlace()

    @parameterized.expand(
        [
            # regular convolution
            [8, 4, 2, (16, 32), (16, 32), [2], "equiangular", "equiangular", False, 1e-5],
            [8, 4, 2, (16, 32), (8, 16), [3], "equiangular", "equiangular", False, 1e-5],
            [8, 4, 2, (16, 32), (8, 16), [2, 3], "equiangular", "equiangular", False, 1e-5],
            [8, 4, 2, (18, 36), (6, 12), [4], "equiangular", "equiangular", False, 1e-5],
            [8, 4, 2, (16, 32), (8, 16), [3], "equiangular", "legendre-gauss", False, 1e-5],
            [8, 4, 2, (16, 32), (8, 16), [3], "legendre-gauss", "equiangular", False, 1e-5],
            [8, 4, 2, (16, 32), (8, 16), [3], "legendre-gauss", "legendre-gauss", False, 1e-5],
            # transpose convolution
            [8, 4, 2, (16, 32), (16, 32), [2], "equiangular", "equiangular", True, 1e-5],
            [8, 4, 2, (8, 16), (16, 32), [3], "equiangular", "equiangular", True, 1e-5],
            [8, 4, 2, (8, 16), (16, 32), [2, 3], "equiangular", "equiangular", True, 1e-5],
            [8, 4, 2, (6, 12), (18, 36), [4], "equiangular", "equiangular", True, 1e-5],
            [8, 4, 2, (8, 16), (16, 32), [3], "equiangular", "legendre-gauss", True, 1e-5],
            [8, 4, 2, (8, 16), (16, 32), [3], "legendre-gauss", "equiangular", True, 1e-5],
            [8, 4, 2, (8, 16), (16, 32), [3], "legendre-gauss", "legendre-gauss", True, 1e-5],
        ]
    )
    def test_disco_convolution(
        self,
        batch_size,
        in_channels,
        out_channels,
        in_shape,
        out_shape,
        kernel_shape,
        grid_in,
        grid_out,
        transpose,
        tol,
    ):
        Conv = DiscreteContinuousConvTransposeS2 if transpose else DiscreteContinuousConvS2
        conv = Conv(
            in_channels,
            out_channels,
            in_shape,
            out_shape,
            kernel_shape,
            groups=1,
            grid_in=grid_in,
            grid_out=grid_out,
            bias=False,
        ).to(self.device)

        nlat_in, nlon_in = in_shape
        nlat_out, nlon_out = out_shape

        theta_cutoff = (kernel_shape[0] + 1) * np.pi / float(nlat_in - 1)

        if transpose:
            psi_dense = _precompute_convolution_tensor_dense(
                out_shape,
                in_shape,
                kernel_shape,
                grid_in=grid_out,
                grid_out=grid_in,
                theta_cutoff=theta_cutoff,
            ).to(self.device)
        else:
            psi_dense = _precompute_convolution_tensor_dense(
                in_shape,
                out_shape,
                kernel_shape,
                grid_in=grid_in,
                grid_out=grid_out,
                theta_cutoff=theta_cutoff,
            ).to(self.device)

            psi = paddle.sparse.sparse_coo_tensor(
                conv.psi_idx,
                conv.psi_vals,
                shape=(conv.kernel_size, conv.nlat_out, conv.nlat_in * conv.nlon_in),
            ).to_dense()

            self.assertTrue(
                paddle.allclose(psi, psi_dense[:, :, 0].reshape(-1, nlat_out, nlat_in * nlon_in))
            )

        # create a copy of the weight
        w_ref = conv.weight.detach().clone()
        w_ref.stop_gradient = not True

        # create an input signal
        paddle.seed(seed=333)
        x = paddle.randn(shape=[batch_size, in_channels, *in_shape])
        x.stop_gradient = not True
        x = x.to(self.device)

        # perform the reference computation
        x_ref = x.clone().detach()
        x_ref.stop_gradient = not True
        if transpose:
            y_ref = paddle.einsum("oif,biqr->bofqr", w_ref, x_ref)
            y_ref = paddle.einsum("fqrtp,bofqr->botp", psi_dense, y_ref * conv.quad_weights)
        else:
            y_ref = paddle.einsum("ftpqr,bcqr->bcftp", psi_dense, x_ref * conv.quad_weights)
            y_ref = paddle.einsum("oif,biftp->botp", w_ref, y_ref)

        # use the convolution module
        y = conv(x)

        # compare results
        self.assertTrue(paddle.allclose(y, y_ref, rtol=tol, atol=tol))

        # compute gradients and compare results
        grad_input = paddle.randn(shape=y.shape, dtype=y.dtype)
        y_ref.backward(grad_input)
        y.backward(grad_input)

        # compare
        self.assertTrue(paddle.allclose(x.grad, x_ref.grad, rtol=tol, atol=tol))
        self.assertTrue(paddle.allclose(conv.weight.grad, w_ref.grad, rtol=tol, atol=tol))


if __name__ == "__main__":
    unittest.main()

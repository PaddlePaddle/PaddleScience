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

import numpy as np
import paddle
import paddle.fft
import paddle.nn as nn

from paddle_harmonics.legendre import _precompute_dlegpoly
from paddle_harmonics.legendre import _precompute_legpoly
from paddle_harmonics.quadrature import clenshaw_curtiss_weights
from paddle_harmonics.quadrature import legendre_gauss_weights
from paddle_harmonics.quadrature import lobatto_weights


class RealSHT(nn.Layer):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(
        self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True
    ):
        r"""
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        weights = paddle.to_tensor(w)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = paddle.to_tensor(pct)
        weights = paddle.einsum("mlk,k->mlk", pct, weights)

        # remember quadrature weights
        self.register_buffer("weights", weights, persistable=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        assert x.shape[-2] == self.nlat
        assert x.shape[-1] == self.nlon

        # apply real fft in the longitudinal direction
        x = 2.0 * np.pi * paddle.fft.rfft(x, axis=-1, norm="forward")

        # do the Legendre-Gauss quadrature
        x = paddle.as_real(x)

        # distributed contraction: fork
        out_shape = list(x.shape)
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = paddle.zeros(out_shape, dtype=x.dtype)

        # contraction
        xout[..., 0] = paddle.einsum(
            "...km,mlk->...lm", x[..., : self.mmax, 0], self.weights.astype(x.dtype)
        )
        xout[..., 1] = paddle.einsum(
            "...km,mlk->...lm", x[..., : self.mmax, 1], self.weights.astype(x.dtype)
        )
        x = paddle.as_complex(xout)

        return x


class InverseRealSHT(nn.Layer):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(
        self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True
    ):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        pct = _precompute_legpoly(
            self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase
        )
        pct = paddle.to_tensor(pct)

        # register buffer
        self.register_buffer("pct", pct, persistable=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        assert x.shape[-2] == self.lmax
        assert x.shape[-1] == self.mmax

        # Evaluate associated Legendre functions on the output nodes
        x = paddle.as_real(x)

        rl = paddle.einsum("...lm, mlk->...km", x[..., 0], self.pct.astype(x.dtype))
        im = paddle.einsum("...lm, mlk->...km", x[..., 1], self.pct.astype(x.dtype))
        xs = paddle.stack((rl, im), -1)

        # apply the inverse (real) FFT
        x = paddle.as_complex(xs)
        x = paddle.fft.irfft(x, n=self.nlon, axis=-1, norm="forward")

        return x


class RealVectorSHT(nn.Layer):
    r"""
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(
        self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True
    ):
        r"""
        Initializes the vector SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        weights = paddle.to_tensor(w)
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        dpct = paddle.to_tensor(dpct)

        # combine integration weights, normalization factor in to one:
        l = paddle.arange(0, self.lmax).astype(dpct.dtype)
        norm_factor = 1.0 / l / (l + 1)
        norm_factor[0] = 1.0
        weights = paddle.einsum("dmlk,k,l->dmlk", dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # remember quadrature weights
        self.register_buffer("weights", weights, persistable=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        assert len(x.shape) >= 3

        # apply real fft in the longitudinal direction
        x = 2.0 * np.pi * paddle.fft.rfft(x, axis=-1, norm="forward")

        # do the Legendre-Gauss quadrature
        x = paddle.as_real(x)

        # distributed contraction: fork
        out_shape = list(x.shape)
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = paddle.zeros(out_shape, dtype=x.dtype)

        # contraction - spheroidal component
        # real component
        xout[..., 0, :, :, 0] = paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, : self.mmax, 0], self.weights[0].to(x.dtype)
        ) - paddle.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 1], self.weights[1].to(x.dtype)
        )

        # iamg component
        xout[..., 0, :, :, 1] = paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, : self.mmax, 1], self.weights[0].to(x.dtype)
        ) + paddle.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 0], self.weights[1].to(x.dtype)
        )

        # contraction - toroidal component
        # real component
        xout[..., 1, :, :, 0] = -paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, : self.mmax, 1], self.weights[1].to(x.dtype)
        ) - paddle.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 0], self.weights[0].to(x.dtype)
        )
        # imag component
        xout[..., 1, :, :, 1] = paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, : self.mmax, 0], self.weights[1].to(x.dtype)
        ) - paddle.einsum(
            "...km,mlk->...lm", x[..., 1, :, : self.mmax, 1], self.weights[0].to(x.dtype)
        )

        return paddle.as_complex(xout)


class InverseRealVectorSHT(nn.Layer):
    r"""
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(
        self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True
    ):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat - 1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        dpct = _precompute_dlegpoly(
            self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase
        )
        dpct = paddle.to_tensor(dpct)

        # register weights
        self.register_buffer("dpct", dpct, persistable=False)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        assert x.shape[-2] == self.lmax
        assert x.shape[-1] == self.mmax

        # Evaluate associated Legendre functions on the output nodes
        x = paddle.as_real(x)

        # contraction - spheroidal component
        # real component
        srl = paddle.einsum(
            "...lm,mlk->...km", x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)
        ) - paddle.einsum("...lm,mlk->...km", x[..., 1, :, :, 1], self.dpct[1].to(x.dtype))
        # iamg component
        sim = paddle.einsum(
            "...lm,mlk->...km", x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)
        ) + paddle.einsum("...lm,mlk->...km", x[..., 1, :, :, 0], self.dpct[1].to(x.dtype))

        # contraction - toroidal component
        # real component
        trl = -paddle.einsum(
            "...lm,mlk->...km", x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)
        ) - paddle.einsum("...lm,mlk->...km", x[..., 1, :, :, 0], self.dpct[0].to(x.dtype))
        # imag component
        tim = paddle.einsum(
            "...lm,mlk->...km", x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)
        ) - paddle.einsum("...lm,mlk->...km", x[..., 1, :, :, 1], self.dpct[0].to(x.dtype))

        # reassemble
        s = paddle.stack((srl, sim), -1)
        t = paddle.stack((trl, tim), -1)
        xs = paddle.stack((s, t), -4)

        # apply the inverse (real) FFT
        x = paddle.as_complex(xs)
        x = paddle.fft.irfft(x, n=self.nlon, axis=-1, norm="forward")

        return x

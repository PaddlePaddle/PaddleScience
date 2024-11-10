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
import paddle.nn.functional as F  # noqa

from paddle_harmonics.distributed import azimuth_group_rank
from paddle_harmonics.distributed import azimuth_group_size
from paddle_harmonics.distributed import compute_split_shapes
from paddle_harmonics.distributed import distributed_transpose_azimuth
from paddle_harmonics.distributed import distributed_transpose_polar
from paddle_harmonics.distributed import polar_group_rank
from paddle_harmonics.distributed import polar_group_size
from paddle_harmonics.distributed import split_tensor_along_dim
from paddle_harmonics.legendre import _precompute_dlegpoly
from paddle_harmonics.legendre import _precompute_legpoly
from paddle_harmonics.quadrature import clenshaw_curtiss_weights
from paddle_harmonics.quadrature import legendre_gauss_weights
from paddle_harmonics.quadrature import lobatto_weights


class DistributedRealSHT(nn.Layer):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(
        self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True
    ):
        """
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

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # combine quadrature weights with the legendre weights
        weights = paddle.to_tensor(w)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = paddle.to_tensor(pct)
        weights = paddle.einsum("mlk,k->mlk", pct, weights)

        # split weights
        weights = split_tensor_along_dim(weights, dim=0, num_chunks=self.comm_size_azimuth)[
            self.comm_rank_azimuth
        ]

        # remember quadrature weights
        self.register_buffer("weights", weights, persistable=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * np.pi * paddle.fft.rfft(x, n=self.nlon, axis=-1, norm="forward")

        # truncate
        x = x[..., : self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = paddle.as_real(x)

        # contraction
        xs = paddle.einsum("...kmr,mlk->...lmr", x, self.weights.to(x.dtype)).contiguous()

        # cast to complex
        x = paddle.as_complex(xs)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        return x


class DistributedInverseRealSHT(nn.Layer):
    """
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

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute legende polynomials
        pct = _precompute_legpoly(
            self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase
        )
        pct = paddle.to_tensor(pct)

        # split in m
        pct = split_tensor_along_dim(pct, dim=0, num_chunks=self.comm_size_azimuth)[
            self.comm_rank_azimuth
        ]

        # register
        self.register_buffer("pct", pct, persistable=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[1]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = paddle.as_real(x)

        # einsum
        xs = paddle.einsum("...lmr, mlk->...kmr", x, self.pct.to(x.dtype)).contiguous()
        # rl = paddle.einsum('...lm, mlk->...km', x[..., 0], self.pct.to(x.dtype) )
        # im = paddle.einsum('...lm, mlk->...km', x[..., 1], self.pct.to(x.dtype) )
        # xs = paddle.stack((rl, im), -1).contiguous()

        # inverse FFT
        x = paddle.as_complex(xs)

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = paddle.fft.irfft(x, n=self.nlon, axis=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        return x


class DistributedRealVectorSHT(nn.Layer):
    """
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(
        self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True
    ):
        """
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

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute weights
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

        # we need to split in m, pad before:
        weights = split_tensor_along_dim(weights, dim=1, num_chunks=self.comm_size_azimuth)[
            self.comm_rank_azimuth
        ]

        # remember quadrature weights
        self.register_buffer("weights", weights, persistable=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        assert len(x.shape) >= 3

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * np.pi * paddle.fft.rfft(x, n=self.nlon, axis=-1, norm="forward")

        # truncate
        x = x[..., : self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = paddle.as_real(x)

        # create output array
        xs = paddle.zeros_like(x, dtype=x.dtype)

        # contraction - spheroidal component
        # real component
        xs[..., 0, :, :, 0] = paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, :, 0], self.weights[0].to(xs.dtype)
        ) - paddle.einsum("...km,mlk->...lm", x[..., 1, :, :, 1], self.weights[1].to(xs.dtype))
        # imag component
        xs[..., 0, :, :, 1] = paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, :, 1], self.weights[0].to(xs.dtype)
        ) + paddle.einsum("...km,mlk->...lm", x[..., 1, :, :, 0], self.weights[1].to(xs.dtype))

        # contraction - toroidal component
        # real component
        xs[..., 1, :, :, 0] = -paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, :, 1], self.weights[1].to(xs.dtype)
        ) - paddle.einsum("...km,mlk->...lm", x[..., 1, :, :, 0], self.weights[0].to(xs.dtype))
        # imag component
        xs[..., 1, :, :, 1] = paddle.einsum(
            "...km,mlk->...lm", x[..., 0, :, :, 0], self.weights[1].to(xs.dtype)
        ) - paddle.einsum("...km,mlk->...lm", x[..., 1, :, :, 1], self.weights[0].to(xs.dtype))

        # pad if required
        x = paddle.as_complex(xs)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        return x


class DistributedInverseRealVectorSHT(nn.Layer):
    """
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

        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute legende polynomials
        dpct = _precompute_dlegpoly(
            self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase
        )
        dpct = paddle.to_tensor(dpct)

        # split in m
        dpct = split_tensor_along_dim(dpct, dim=1, num_chunks=self.comm_size_azimuth)[
            self.comm_rank_azimuth
        ]

        # register buffer
        self.register_buffer("dpct", dpct, persistable=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f"nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}"

    def forward(self, x: paddle.Tensor):

        # store num channels
        num_chans = x.shape[1]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = paddle.as_real(x)

        # contraction - spheroidal component
        # real component
        srl = paddle.einsum(
            "...lm,mlk->...km", x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)
        ) - paddle.einsum("...lm,mlk->...km", x[..., 1, :, :, 1], self.dpct[1].to(x.dtype))
        # imag component
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

        # convert to complex
        x = paddle.as_complex(xs)

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = paddle.fft.irfft(x, n=self.nlon, axis=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        return x

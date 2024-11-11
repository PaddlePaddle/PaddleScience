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


import paddle

BLOCK_SIZE_BATCH = 4
BLOCK_SIZE_NZ = 8
BLOCK_SIZE_POUT = 8


def _disco_s2_contraction_paddle(x: paddle.Tensor, psi: paddle.Tensor, nlon_out: int):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in Triton.
    """
    assert len(psi.shape) == 3
    assert len(x.shape) == 4

    batch_size, n_chans, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, _ = psi.shape

    assert psi.shape[-1] == nlat_in * nlon_in
    assert nlon_in % nlon_out == 0
    assert nlon_in >= nlat_out
    pscale = nlon_in // nlon_out

    # add a dummy dimension for nkernel and move the batch and channel dims to the end
    x = x.reshape([1, batch_size * n_chans, nlat_in, nlon_in]).transpose([0, 2, 3, 1])
    x = x.expand([kernel_size, -1, -1, -1])

    y = paddle.zeros(
        [nlon_out, kernel_size, nlat_out, batch_size * n_chans],
        dtype=x.dtype,
    ).to(x.place)

    for pout in range(nlon_out):
        # sparse contraction with psi
        y[pout] = paddle.bmm(
            psi.to_dense(), x.reshape([kernel_size, nlat_in * nlon_in, -1])
        )
        # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
        x = paddle.roll(x, -pscale, axis=2)

    # reshape y back to expose the correct dimensions
    y = y.transpose([3, 1, 2, 0]).reshape(
        [batch_size, n_chans, kernel_size, nlat_out, nlon_out]
    )

    return y


def _disco_s2_transpose_contraction_paddle(
    x: paddle.Tensor, psi: paddle.Tensor, nlon_out: int
):
    """
    Reference implementation of the custom contraction as described in [1]. This requires repeated
    shifting of the input tensor, which can potentially be costly. For an efficient implementation
    on GPU, make sure to use the custom kernel written in Triton.
    """
    assert len(psi.shape) == 3
    assert len(x.shape) == 5

    batch_size, n_chans, kernel_size, nlat_in, nlon_in = x.shape
    kernel_size, _, n_out = psi.shape

    assert psi.shape[-2] == nlat_in
    assert n_out % nlon_out == 0
    nlat_out = n_out // nlon_out
    assert nlon_out >= nlat_in
    pscale = nlon_out // nlon_in

    # we do a semi-transposition to faciliate the computation
    inz = psi.indices()
    tout = inz[2] // nlon_out
    pout = inz[2] % nlon_out
    # flip the axis of longitudes
    pout = nlon_out - 1 - pout
    tin = inz[1]
    inz = paddle.stack([inz[0], tout, tin * nlon_out + pout], axis=0)
    psi_mod = paddle.sparse.sparse_coo_tensor(
        inz, psi.values(), shape=(kernel_size, nlat_out, nlat_in * nlon_out)
    )

    # interleave zeros along the longitude dimension to allow for fractional offsets to be considered
    x_ext = paddle.zeros(
        [kernel_size, nlat_in, nlon_out, batch_size * n_chans],
        dtype=x.dtype,
    ).to(x.place)
    x_ext[:, :, ::pscale, :] = x.reshape(
        [batch_size * n_chans, kernel_size, nlat_in, nlon_in]
    ).transpose([1, 2, 3, 0])
    # we need to go backwards through the vector, so we flip the axis
    x_ext = x_ext.contiguous()

    y = paddle.zeros(
        [kernel_size, nlon_out, nlat_out, batch_size * n_chans],
        dtype=x.dtype,
    ).to(x.place)

    for pout in range(nlon_out):
        # we need to repeatedly roll the input tensor to faciliate the shifted multiplication
        # TODO: double-check why this has to happen first
        x_ext = paddle.roll(x_ext, -1, axis=2)
        # sparse contraction with the modified psi
        y[:, pout, :, :] = paddle.bmm(
            psi_mod.to_dense(), x_ext.reshape([kernel_size, nlat_in * nlon_out, -1])
        )

    # sum over the kernel dimension and reshape to the correct output size
    y = (
        y.sum(axis=0)
        .transpose([2, 1, 0])
        .reshape([batch_size, n_chans, nlat_out, nlon_out])
    )

    return y

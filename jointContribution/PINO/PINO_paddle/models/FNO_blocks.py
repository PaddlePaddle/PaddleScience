import itertools

import paddle
import paddle.nn as nn

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    # order = tl.ndim(x)
    order = len(x.shape)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)
    # For the darcy flow, the only einsum is abcd,becd->aecd, where x and weights are shaped [32,32,8,8]
    if not isinstance(weight, paddle.Tensor):
        weight = paddle.to_tensor(weight)

    return paddle.einsum(eq, x, weight)


def _contract_dense_trick(x, weight, separable=False):
    # the same as above function, but do the complex multiplication manually to avoid the einsum bug in paddle
    weight_real = weight.data.real()
    weight_imag = weight.data.imag()

    order = len(x.shape)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)

    o1_real = paddle.einsum(eq, x.real(), weight_real) - paddle.einsum(
        eq, x.imag(), weight_imag
    )
    o1_imag = paddle.einsum(eq, x.imag(), weight_real) + paddle.einsum(
        eq, x.real(), weight_imag
    )
    x = paddle.complex(o1_real, o1_imag)
    return x


def _contract_dense_separable(x, weight, separable=True):
    if separable == False:
        raise ValueError("This function is only for separable=True")
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    # order = tl.ndim(x)
    order = len(x.shape)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = (
        x_syms + "," + rank_sym + "," + ",".join(factor_syms) + "->" + "".join(out_syms)
    )

    return paddle.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    # order = tl.ndim(x)
    order = len(x.shape)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...

    eq = (
        x_syms
        + ","
        + core_syms
        + ","
        + ",".join(factor_syms)
        + "->"
        + "".join(out_syms)
    )
    print(eq)  # 'abcd,fghi,bf,eg,ch,di->aecd'
    return paddle.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    # order = tl.ndim(x)
    order = len(x.shape)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    return paddle.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorl-paddle's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            print("SEPARABLE")
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if isinstance(weight, paddle.Tensor):
            return _contract_dense_trick
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation = {implementation}, expected "reconstructed" or "factorized"'
        )


class FactorizedTensor(nn.Layer):
    def __init__(self, shape, init_scale):
        super().__init__()
        self.shape = shape
        self.init_scale = init_scale
        self.real = self.create_parameter(
            shape=shape, default_initializer=nn.initializer.XavierNormal()
        )
        self.imag = self.create_parameter(
            shape=shape, default_initializer=nn.initializer.XavierNormal()
        )

    def __repr__(self):
        return f"FactorizedTensor(shape={self.shape})"

    @property
    def data(self):
        return paddle.complex(self.real, self.imag)


class FactorizedSpectralConv(nn.Layer):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    kept_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    separable : bool, default is True
    scale : float or 'auto', default is 'auto'
        scale to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    factorization : str, {'tucker', 'cp', 'tt'}, optional
        Tensor factorization of the parameters weight to use, by default 'tucker'
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    fft_norm : str, optional
        by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        n_layers=1,
        scale="auto",
        separable=False,
        fft_norm="backward",
        bias=False,
        implementation="reconstructed",
        joint_factorization=False,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=False,
        decomposition_kwargs=dict(),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = len(n_modes)

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_modes is half of that except in the last mode, correponding to the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        half_modes = [m // 2 for m in n_modes]
        self.half_modes = n_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        self.mlp = None

        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal to out_channels, ",
                    f"but got in_channels = {in_channels} and out_channels = {out_channels}",
                )
            weight_shape = (in_channels, *self.half_modes)
        else:
            weight_shape = (in_channels, out_channels, *self.half_modes)
        self.separable = separable

        if joint_factorization:
            self.weight = paddle.create_parameter(
                shape=((2 ** (self.order - 1)) * n_layers, *weight_shape),
                dtype="float32",
            )
        else:
            self.weight = nn.LayerList(
                [
                    FactorizedTensor(weight_shape, init_scale=scale)
                    for _ in range((2 ** (self.order - 1)) * n_layers)
                ]
            )

        self._contract = get_contract_fun(
            self.weight[0].data, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = paddle.create_parameter(
                shape=((n_layers, self.out_channels) + (1,) * self.order),
                dtype="float32",
            )
        else:
            self.bias = None

    def forward(self, x, indices=0):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : paddle.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient

        # Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))

        # put x back in to real, as in paddle x.float()
        x_float = paddle.cast(x, dtype="float32")
        x = paddle.fft.rfftn(x_float, norm=self.fft_norm, axes=fft_dims)

        out_fft = paddle.zeros(
            [batchsize, self.out_channels, *fft_size],
            dtype=paddle.complex64,
        )  # [1,32,16,9], all zeros, complex

        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [
            ((None, self.half_modes[-1]),)
        ]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            if len(idx_tuple) == 4:
                out_fft[
                    idx_tuple[0], idx_tuple[1], idx_tuple[2], idx_tuple[3]
                ] = self._contract(
                    x[idx_tuple[0], idx_tuple[1], idx_tuple[2], idx_tuple[3]],
                    self.weight[indices + i].real,
                    self.weight[indices + i].imag,
                    separable=self.separable,
                )
            elif len(idx_tuple) == 3:
                out_fft[idx_tuple[0], idx_tuple[1], idx_tuple[2]] = self._contract(
                    x[idx_tuple[0], idx_tuple[1], idx_tuple[2]],
                    self.weight[indices + i].real,
                    self.weight[indices + i].imag,
                    separable=self.separable,
                )
            else:
                raise ValueError("Not implemented")

        x = paddle.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv2d(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class SubConv2d(nn.Layer):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x):
        return self.main_conv.forward(x, self.indices)


class FactorizedSpectralConv1d(FactorizedSpectralConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_height,
        n_layers=1,
        scale="auto",
        separable=False,
        fft_norm="backward",
        bias=True,
        implementation="reconstucted",
        joint_factorization=False,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=False,
        decomposition_kwargs=dict(),
    ):
        super().__init__(
            in_channels,
            out_channels,
            (modes_height,),
            n_layers=n_layers,
            scale=scale,
            separable=separable,
            fft_norm=fft_norm,
            bias=bias,
            implementation=implementation,
            joint_factorization=joint_factorization,
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs,
        )
        self.half_modes_height = self.half_modes[0]

    def forward(self, x, indices=0):
        batchsize, channels, width = x.shape

        x = paddle.fft.rfft(x, norm=self.fft_norm)

        out_fft = paddle.zeros(
            [batchsize, self.out_channels, width // 2 + 1],
            dtype=paddle.complex64,
        )
        out_fft[:, :, : self.half_modes_height] = self._contract(
            x[:, :, : self.half_modes_height],
            self.weight[indices],
            separable=self.separable,
        )

        x = paddle.fft.irfft(out_fft, n=width, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FactorizedSpectralConv2d(FactorizedSpectralConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_height,
        modes_width,
        n_layers=1,
        scale="auto",
        separable=False,
        fft_norm="backward",
        bias=False,
        implementation="factorized",
        joint_factorization=False,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=False,
        decomposition_kwargs=dict(),
    ):
        super().__init__(
            in_channels,
            out_channels,
            (modes_height, modes_width),
            n_layers=n_layers,
            scale=scale,
            separable=separable,
            fft_norm=fft_norm,
            bias=bias,
            implementation=implementation,
            joint_factorization=joint_factorization,
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs,
        )
        self.half_modes_height, self.half_modes_width = self.half_modes

    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x_float = paddle.cast(x, dtype="float32")
        x = paddle.fft.rfft2(x_float, norm=self.fft_norm)

        # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_fft = paddle.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
        )

        # upper block (truncate high freq)
        out_fft[
            :, :, : self.half_modes_height, : self.half_modes_width
        ] = self._contract(
            x[:, :, : self.half_modes_height, : self.half_modes_width],
            self.weight[2 * indices],
            separable=self.separable,
        )
        # Lower block
        out_fft[
            :, :, -self.half_modes_height :, : self.half_modes_width
        ] = self._contract(
            x[:, :, -self.half_modes_height :, : self.half_modes_width],
            self.weight[2 * indices + 1],
            separable=self.separable,
        )

        x = paddle.fft.irfft2(
            out_fft, s=(height, width), axes=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FactorizedSpectralConv3d(FactorizedSpectralConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_height,
        modes_width,
        modes_depth,
        n_layers=1,
        scale="auto",
        separable=False,
        fft_norm="backward",
        bias=True,
        implementation="reconstucted",
        joint_factorization=False,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=False,
        decomposition_kwargs=dict(),
    ):
        super().__init__(
            in_channels,
            out_channels,
            (modes_height, modes_width, modes_depth),
            n_layers=n_layers,
            scale=scale,
            separable=separable,
            fft_norm=fft_norm,
            bias=bias,
            implementation=implementation,
            joint_factorization=joint_factorization,
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            decomposition_kwargs=decomposition_kwargs,
        )
        (
            self.half_modes_height,
            self.half_modes_width,
            self.half_modes_depth,
        ) = self.half_modes

    def forward(self, x, indices=0):
        batchsize, channels, height, width, depth = x.shape

        x_float = paddle.cast(x, dtype="float32")
        x = paddle.fft.rfftn(x_float, norm=self.fft_norm, dim=[-3, -2, -1])
        out_fft = paddle.zeros(
            [batchsize, self.out_channels, height, width, depth // 2 + 1],
            dtype="complex64",
        )

        out_fft[
            :,
            :,
            : self.half_modes_height,
            : self.half_modes_width,
            : self.half_modes_depth,
        ] = self._contract(
            x[
                :,
                :,
                : self.half_modes_height,
                : self.half_modes_width,
                : self.half_modes_depth,
            ],
            self.weight[indices + 0],
            separable=self.separable,
        )
        out_fft[
            :,
            :,
            : self.half_modes_height,
            -self.half_modes_width :,
            : self.half_modes_depth,
        ] = self._contract(
            x[
                :,
                :,
                : self.half_modes_height,
                -self.half_modes_width :,
                : self.half_modes_depth,
            ],
            self.weight[indices + 1],
            separable=self.separable,
        )
        out_fft[
            :,
            :,
            -self.half_modes_height :,
            : self.half_modes_width,
            : self.half_modes_depth,
        ] = self._contract(
            x[
                :,
                :,
                -self.half_modes_height :,
                : self.half_modes_width,
                : self.half_modes_depth,
            ],
            self.weight[indices + 2],
            separable=self.separable,
        )
        out_fft[
            :,
            :,
            -self.half_modes_height :,
            -self.half_modes_width :,
            : self.half_modes_depth,
        ] = self._contract(
            x[
                :,
                :,
                -self.half_modes_height :,
                -self.half_modes_width :,
                : self.half_modes_depth,
            ],
            self.weight[indices + 3],
            separable=self.separable,
        )

        x = paddle.fft.irfftn(out_fft, s=(height, width, depth), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


if __name__ == "__main__":
    # let x be a complex tensor of size (32, 32, 8, 8)
    x = paddle.randn([32, 32, 8, 8]).astype("complex64")
    # let weight be the same
    weight = paddle.randn([32, 32, 8, 8]).astype("complex64")
    separable = False
    result = _contract_dense(x, weight, separable=separable)
    print(result)

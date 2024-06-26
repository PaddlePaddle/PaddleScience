from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
import paddle.nn.functional as F
from paddle import nn

from ppsci.arch import base
from ppsci.arch import fno_block
from ppsci.arch.paddle_harmonics import sht as paddle_sht
from ppsci.utils import initializer

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False, dhconv=True):
    order = len(x.shape)
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

    if dhconv:
        weight_syms.pop()

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)
    # For the darcy flow, the only einsum is abcd,becd->aecd, where x and weights are shaped [32,32,8,8]
    if not isinstance(weight, paddle.Tensor):
        weight = paddle.to_tensor(weight)

    return paddle.einsum(eq, x, weight)


def _contract_dense_trick(x, weight_real, weight_imag, separable=False, dhconv=True):
    # the same as above function, but do the complex multiplication manually to avoid the einsum bug in paddle
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

    if dhconv:
        weight_syms.pop()

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
    if not separable:
        raise ValueError("This function is only for separable=True")
    return x * weight


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction.

    Args:
        weight (FactorizedTensor): The factoriz Tensor.
        implementation (str, optional): Whether to reconstruct the weight and do a forward pass (reconstructed)
            or contract directly the factors of the factorized weight with the input (factorized).
            {'reconstructed', 'factorized'} Defaults to "reconstructed".
        separable (bool, optional): Whether to use the separable implementation of contraction. This arg is
            only checked when `implementation=reconstructed`. Defaults to False.
    """

    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense_trick
    elif implementation == "factorized":
        if isinstance(weight, paddle.Tensor):
            return _contract_dense_trick

    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


class SHT(nn.Layer):
    """A wrapper for the Spherical Harmonics transform

    Allows to call it with an interface similar to that of FFT
    """

    def __init__(self, dtype=paddle.float32):
        super().__init__()
        self.dtype = dtype
        self._SHT_cache = nn.LayerDict()
        self._iSHT_cache = nn.LayerDict()

    def sht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, height, width = x.shape  # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                modes_width = height // 2
            else:
                modes_width = height
            modes_height = height
        else:
            modes_height, modes_width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            sht = self._SHT_cache[cache_key]
        except KeyError:
            sht = paddle_sht.RealSHT(
                nlat=height,
                nlon=width,
                lmax=modes_height,
                mmax=modes_width,
                grid=grid,
                norm=norm,
            ).astype(dtype=self.dtype)

            self._SHT_cache[cache_key] = sht

        return sht(x)

    def isht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, modes_height, modes_width = x.shape  # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                width = modes_width * 2
            else:
                width = modes_width
            height = modes_height
        else:
            height, width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            isht = self._iSHT_cache[cache_key]
        except KeyError:
            isht = paddle_sht.InverseRealSHT(
                nlat=height,
                nlon=width,
                lmax=modes_height,
                mmax=modes_width,
                grid=grid,
                norm=norm,
            ).astype(dtype=self.dtype)
            self._iSHT_cache[cache_key] = isht

        return isht(x)


Number = Union[int, float]


class SphericalConv(nn.Layer):
    """Spherical Convolution, base class for the SFNO [1].
        .. [1] Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere,
           Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, Anima Anandkumar,
           ICML 2023.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_modes (Tuple[int, ...]): Number of modes to use for contraction in Fourier domain during
            training.
        max_n_modes (int, optional): The maximum number of modes to use for contraction in Fourier domain during
            training. Defaults to None.
        bias (bool, optional): Whether to use bias in the layers. Defaults to True.
        n_layers (int, optional): Number of Fourier Layers. Defaults to 1.
        separable (bool, optional): Whether to use separable Fourier Conv. Defaults to False.
        output_scaling_factor (Optional[Union[Number, List[Number]]], optional):  Scaling factor for the
            output. Defaults to None.
        rank (float, optional):  Rank of the tensor factorization of the Fourier weights. Defaults to 0.5.
        factorization (str, optional): Tensor factorization of the parameters weight to use. Defaults to "dense".
        implementation (str, optional): If factorization is not None, forward mode to use. Defaults to "reconstructed".
        joint_factorization (bool, optional):  Whether all the Fourier Layers should be parametrized by a
            single tensor. Defaults to False.
        init_std (str, optional): The std to use for the init. Defaults to "auto".
        sht_norm (str, optional): The normalization mode of the SHT. Defaults to "ortho".
        sht_grids (str, optional): The grid of the SHT. Defaults to "equiangular".
        dtype (paddle.float32, optional): The data type. Defaults to paddle.float32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        max_n_modes: int = None,
        bias: bool = True,
        n_layers: int = 1,
        separable: bool = False,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        rank: float = 0.5,
        factorization: str = "dense",
        implementation: str = "reconstructed",
        joint_factorization: bool = False,
        init_std: str = "auto",
        sht_norm: str = "ortho",
        sht_grids: str = "equiangular",
        dtype: paddle.dtype = paddle.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dtype = dtype

        self.joint_factorization = joint_factorization

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.order = len(n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = fno_block.validate_scaling_factor(
            output_scaling_factor, self.order, n_layers
        )

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    f"To use separable Fourier Conv, in_channels must be equal to out_channels, but got in_channels={in_channels} and out_channels={out_channels}"
                )
            weight_shape = (in_channels, *self.n_modes[:-1])
        else:
            weight_shape = (in_channels, out_channels, *self.n_modes[:-1])
        self.separable = separable

        if joint_factorization:
            self.weight = paddle.create_parameter(
                shape=(n_layers, *weight_shape),
                dtype="float32",
            )
            self.weight = initializer.normal_(self.weight, 0, init_std)
        else:
            self.weight = nn.LayerList(
                [
                    fno_block.FactorizedTensor(weight_shape, init_scale=init_std)
                    for _ in range(n_layers)
                ]
            )
        self._contract = get_contract_fun(
            self.weight[0].data, implementation=implementation, separable=separable
        )
        if bias:
            shape = (n_layers, self.out_channels) + (1,) * self.order
            init_bias = init_std * paddle.randn(shape)
            self.bias = paddle.create_parameter(
                shape=shape,
                dtype=(init_bias.dtype),
                default_initializer=nn.initializer.Assign(init_bias),
            )
            self.bias.stop_gradient = False
        else:
            self.bias = None

        self.sht_norm = sht_norm
        if isinstance(sht_grids, str):
            sht_grids = [sht_grids] * (self.n_layers + 1)
        self.sht_grids = sht_grids
        self.sht_handle = SHT(dtype=self.dtype)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        self._n_modes = n_modes

    def forward(self, x, indices=0, output_shape=None):
        batchsize, channels, height, width = x.shape

        if self.output_scaling_factor is not None and output_shape is None:
            scaling_factors = self.output_scaling_factor[indices]
            height = round(height * scaling_factors[0])
            width = round(width * scaling_factors[1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]

        out_fft = self.sht_handle.sht(
            x,
            s=(self.n_modes[0], self.n_modes[1] // 2),
            norm=self.sht_norm,
            grid=self.sht_grids[indices],
        )

        w_real = self.weight[indices].real[:, :, : self.n_modes[0]]
        w_imag = self.weight[indices].imag[:, :, : self.n_modes[0]]

        out_fft = self._contract(
            out_fft[:, :, : self.n_modes[0], : self.n_modes[1] // 2],
            w_real,
            w_imag,
            separable=self.separable,
            dhconv=True,
        )

        x = self.sht_handle.isht(
            out_fft,
            s=(height, width),
            norm=self.sht_norm,
            grid=self.sht_grids[indices + 1],
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def transform(self, x, layer_index=0, output_shape=None):
        *_, in_height, in_width = x.shape

        if self.output_scaling_factor is not None and output_shape is None:
            height = round(in_height * self.output_scaling_factor[layer_index][0])
            width = round(in_width * self.output_scaling_factor[layer_index][1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]
        else:
            height, width = in_height, in_width

        # Return the identity if the resolution and grid of the input and output are the same
        if ((in_height, in_width) == (height, width)) and (
            self.sht_grids[layer_index] == self.sht_grids[layer_index + 1]
        ):
            return x
        else:
            coefs = self.sht_handle.sht(
                x, s=self.n_modes, norm=self.sht_norm, grid=self.sht_grids[layer_index]
            )
            return self.sht_handle.isht(
                coefs,
                s=(height, width),
                norm=self.sht_norm,
                grid=self.sht_grids[layer_index + 1],
            )


class SFNONet(base.Arch):
    """N-Dimensional Tensorized Fourier Neural Operator.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        n_modes (Tuple[int, ...]): Number of modes to keep in Fourier Layer, along each dimension
            The dimensionality of the SFNO is inferred from ``len(n_modes)`
        hidden_channels (int): Width of the FNO (i.e. number of channels)
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        lifting_channels (int, optional): Number of hidden channels of the lifting block of the FNO.
            Defaults to 256.
        projection_channels (int, optional): Number of hidden channels of the projection block of the FNO.
            Defaults to 256.
        n_layers (int, optional): Number of Fourier Layers. Defaults to 4.
        use_mlp (bool, optional): Whether to use an MLP layer after each FNO block. Defaults to False.
        mlp (Dict[str, float], optional): Parameters of the MLP. {'expansion': float, 'dropout': float}.
            Defaults to None.
        non_linearity (nn.functional, optional): Non-Linearity module to use. Defaults to F.gelu.
        norm (str, optional): Normalization layer to use. Defaults to None.
        ada_in_features (int,optional): The input channles of the adaptive normalization.Defaults to None.
        preactivation (bool, optional): Whether to use resnet-style preactivation. Defaults to False.
        fno_skip (str, optional): Type of skip connection to use,{'linear', 'identity', 'soft-gating'}.
            Defaults to "soft-gating".
        separable (bool, optional): Whether to use a depthwise separable spectral convolution.
            Defaults to  False.
        factorization (str, optional): Tensor factorization of the parameters weight to use.
            * If None, a dense tensor parametrizes the Spectral convolutions.
            * Otherwise, the specified tensor factorization is used. Defaults to "Tucker".
        rank (float, optional): Rank of the tensor factorization of the Fourier weights. Defaults to 1.0.
        joint_factorization (bool, optional): Whether all the Fourier Layers should be parametrized by a
            single tensor (vs one per layer). Defaults to False.
        implementation (str, optional): {'factorized', 'reconstructed'}, optional. Defaults to "factorized".
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass.
            * `factorized` : the input is directly contracted with the factors of the decomposition.
        domain_padding (Optional[list], optional): Whether to use percentage of padding. Defaults to None.
        domain_padding_mode (str, optional): {'symmetric', 'one-sided'}, optional
            How to perform domain padding, by default 'one-sided'. Defaults to "one-sided".
        fft_norm (str, optional): The normalization mode for the FFT. Defaults to "forward".
        patching_levels (int, optional): Number of patching levels to use. Defaults to 0.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        n_modes: Tuple[int, ...],
        hidden_channels: int,
        in_channels: int = 3,
        out_channels: int = 1,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        use_mlp: bool = False,
        mlp: Optional[Dict[str, float]] = None,
        max_n_modes: int = None,
        non_linearity: nn.functional = F.gelu,
        stabilizer: str = None,
        norm: str = None,
        ada_in_features: Optional[int] = None,
        preactivation: bool = False,
        fno_skip: str = "linear",
        mlp_skip: str = "soft-gating",
        separable: bool = False,
        factorization: str = None,
        rank: float = 1.0,
        joint_factorization: bool = False,
        implementation: str = "factorized",
        domain_padding: Optional[list] = None,
        domain_padding_mode: str = "one-sided",
        fft_norm: str = "forward",
        patching_levels: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        if patching_levels:
            self.in_channels = self.in_channels * patching_levels + 1
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fno_skip = (fno_skip,)
        self.mlp_skip = (mlp_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.stabilizer = stabilizer
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = fno_block.DomainPadding(
                domain_padding=domain_padding, padding_mode=domain_padding_mode
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.fno_blocks = fno_block.FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            n_layers=n_layers,
            max_n_modes=max_n_modes,
            use_mlp=use_mlp,
            mlp=mlp,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            ada_in_features=ada_in_features,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            separable=separable,
            factorization=factorization,
            rank=rank,
            SpectralConv=SphericalConv,
            joint_factorization=joint_factorization,
            implementation=implementation,
            fft_norm=fft_norm,
        )
        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = fno_block.MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = fno_block.MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = fno_block.MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x):
        """SFNO's forward pass"""
        x = self.concat_to_tensor(x, self.input_keys)

        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        # x is 0.4 * [1, 32, 16, 16], passed
        for index in range(self.n_layers):
            x = self.fno_blocks(x, index)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        out = self.projection(x)

        return {self.output_keys[0]: out}

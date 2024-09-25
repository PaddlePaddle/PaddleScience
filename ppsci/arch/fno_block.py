import itertools
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import omegaconf
import paddle
import paddle.nn.functional as F
from paddle import nn

from ppsci.utils import initializer
from ppsci.utils import logger

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class DomainPadding(nn.Layer):
    """Applies domain padding scaled automatically to the input's resolution

    Args:
        domain_padding (Union[float, List[float]]): Typically, between zero and one, percentage of padding to use.
        padding_mode (str, optional): Whether to pad on both sides, by default
            'one-sided'.Options are 'symmetric' or 'one-sided'。 Defaults to "one-sided".
        output_scaling_factor (Union[int, List[int]], optional): Scaling factor for the
            output. Defaults to 1.
    """

    def __init__(
        self,
        domain_padding: Union[float, List[float]],
        padding_mode: str = "one-sided",
        output_scaling_factor: Union[int, List[int]] = 1,
    ):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        if output_scaling_factor is None:
            output_scaling_factor = 1
        self.output_scaling_factor: Union[int, List[int]] = output_scaling_factor

        # dict(f'{resolution}'=padding) such that padded = F.pad(x, indices)
        self._padding = dict()

        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        self._unpad_indices = dict()

    def forward(self, x):
        self.pad(x)

    def pad(self, x):
        """Take an input and pad it by the desired fraction

        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)] * len(resolution)

        assert len(self.domain_padding) == len(resolution), (
            "domain_padding length must match the number of spatial/time dimensions "
            "(excluding batch, ch)"
        )

        output_scaling_factor = self.output_scaling_factor
        if not isinstance(self.output_scaling_factor, list):
            # if unset by the user, scaling_factor will be 1 be default,
            # so `output_scaling_factor` should never be None.
            output_scaling_factor: List[float] = validate_scaling_factor(
                self.output_scaling_factor, len(resolution), n_layers=None
            )

        try:
            padding = self._padding[f"{resolution}"]
            return F.pad(x, padding, mode="constant")
        except KeyError:
            padding = [round(p * r) for (p, r) in zip(self.domain_padding, resolution)]

            output_pad = padding
            output_pad = [
                round(i * j) for (i, j) in zip(output_scaling_factor, output_pad)
            ]

            # padding is being applied in reverse order
            # (so we must reverse the padding list)
            padding = padding[::-1]

            # the F.pad(x, padding) funtion pads the tensor 'x' in reverse order of the "padding" list i.e. the last axis of tensor 'x' will be padded by the amount mention at the first position of the 'padding' vector. The details about F.pad can be found here:
            # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/pad_cn.html

            if self.padding_mode == "symmetric":
                # Pad both sides
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_end = None
                        padding_start = None
                    else:
                        padding_end = p
                        padding_start = -p
                    unpad_list.append(slice(padding_end, padding_start, None))

                unpad_indices = (Ellipsis,) + tuple(
                    [slice(p, -p, None) for p in padding]
                )
                padding = [i for p in padding for i in (p, p)]

            elif self.padding_mode == "one-sided":
                # One-side padding
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_start = None
                    else:
                        padding_start = -p
                    unpad_list.append(slice(None, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f"Got self.padding_mode = {self.padding_mode}")

            self._padding[f"{resolution}"] = padding

            padded = F.pad(x, padding, mode="constant")
            output_shape = padded.shape[2:]
            output_shape = [
                round(i * j) for (i, j) in zip(output_scaling_factor, output_shape)
            ]

            self._unpad_indices[f"{[i for i in output_shape]}"] = unpad_indices

            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices[f"{x.shape[2:]}"]

        return x[unpad_indices]


class SoftGating(nn.Layer):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Args:
        in_features (int): The number of input features.
        out_features (int, optional): Number of output features. Defaults to None.
        n_dim (int, optional): Dimensionality of the input (excluding batch-size and channels).
            ``n_dim=2`` corresponds to having Module2D. Defaults to 2.
        bias (bool, optional): Whether to use bias. Defaults to False.
    """

    def __init__(
        self, in_features, out_features: int = None, n_dim: int = 2, bias: bool = False
    ):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features = {in_features} and out_features = {out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features

        self.weight = self.create_parameter(
            shape=(1, self.in_features, *(1,) * n_dim),
            default_initializer=nn.initializer.Constant(1.0),
        )
        if bias:
            self.bias = self.create_parameter(
                shape=(1, self.in_features, *(1,) * n_dim),
                default_initializer=nn.initializer.Constant(1.0),
            )
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


def skip_connection(
    in_features,
    out_features,
    n_dim: int = 2,
    bias: bool = False,
    type: str = "soft-gating",
):
    """A wrapper for several types of skip connections.
       Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        n_dim (int, optional): Dimensionality of the input (excluding batch-size and channels).
            ``n_dim=2`` corresponds to having Module2D. . Defaults to 2.
        bias (bool, optional): Whether to use a bias. Defaults to False.
        type (str, optional): Kind of skip connection to use,{'identity', 'linear', soft-gating'}.
            Defaults to "soft-gating".
    """

    if type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features, out_features=out_features, bias=bias, n_dim=n_dim
        )
    elif type.lower() == "linear":
        return getattr(nn, f"Conv{n_dim}D")(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias_attr=bias,
        )
    elif type.lower() == "identity":
        return nn.Identity()
    else:
        raise ValueError(
            f"Got skip-connection type = {type}, expected one of {'soft-gating', 'linear', 'identity'}."
        )


class AdaIN(nn.Layer):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp

        self.embedding = None

    def set_embedding(self, x):
        self.embedding = x.reshape(
            self.embed_dim,
        )

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"

        weight, bias = paddle.split(
            self.mlp(self.embedding),
            self.embedding.shape[0] // self.in_channels,
            axis=0,
        )

        return nn.functional.group_norm(x, self.in_channels, self.eps, weight, bias)


class MLP(nn.Layer):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Args:
        in_channels (int): The number of input channels.
        out_channels (int, optional): The number of output channels. Defaults to None.
        hidden_channels (int, optional): The number of hidden channels. Defaults to None.
        n_layers (int, optional): The number of layers. Defaults to 2.
        n_dim (int, optional): The type of convolution,2D or 3D. Defaults to 2.
        non_linearity (nn.functional, optional): The activation function. Defaults to F.gelu.
        dropout (float, optional): The ratio of dropout. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        n_layers: int = 2,
        n_dim: int = 2,
        non_linearity: nn.functional = F.gelu,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.LayerList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )

        Conv = getattr(nn, f"Conv{n_dim}D")
        self.fcs = nn.LayerList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x


def _contract_dense(x, weight, separable=False):
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

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)
    # For the darcy flow, the only einsum is abcd,becd->aecd, where x and weights are shaped [32,32,8,8]
    if not isinstance(weight, paddle.Tensor):
        weight = paddle.to_tensor(weight)

    return paddle.einsum(eq, x, weight)


def _contract_dense_trick(x, weight_real, weight_imag, separable=False):
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


def get_contract_fun(
    weight, implementation: str = "reconstructed", separable: bool = False
):
    """Generic ND implementation of Fourier Spectral Conv contraction.

    Args:
        weight (paddle.tensor): FactorizedTensor.
        implementation (str, optional): {'reconstructed', 'factorized'}.
            whether to reconstruct the weight and do a forward pass (reconstructed)
            or contract directly the factors of the factorized weight with the input (factorized). Defaults to "reconstructed".
        separable (bool, optional): Whether to use the separable implementation of contraction. This
            arg  is only checked when `implementation=reconstructed`. Defaults to False.

    Returns:
        function : (x, weight) -> x * weight in Fourier space.
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


Number = Union[float, int]


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """Validates the format and dimensionality of the scaling factor.

    Args:
        scaling_factor (Union[None, Number, List[Number], List[List[Number]]]): The
            scaling factor.
        n_dim (int): The required number of dimensions for expanding `scaling_factor`.
        n_layers (Optional[int], optional): The number of layers for the returned
            nested list. If None, return a single list (rather than a list of lists)
            with `factor` repeated `dim` times. Defaults to None.
    """

    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):

        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all(
            [isinstance(s, (omegaconf.listconfig.ListConfig)) for s in scaling_factor]
        )
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None


def resample(x, res_scale, axis, output_shape=None):
    """A module for generic n-dimentional interpolation (Fourier resampling).

    Args:
        x (paddle.Tensor): Input activation of size (batch_size, channels, d1, ..., dN).
        res_scale (optional[int,tuple]): Scaling factor along each of the dimensions in
            'axis' parameter. If res_scale is scaler, then isotropic scaling is performed.
        axis (int): Axis or dimensions along which interpolation will be performed.
        output_shape (optional[None ,tuple[int]]): The output shape. Defaults to None.
    """

    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale] * len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
            res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(axis), "leght of res_scale and axis are not same"

    old_size = x.shape[-len(axis) :]
    if output_shape is None:
        new_size = tuple([int(round(s * r)) for (s, r) in zip(old_size, res_scale)])
    else:
        new_size = output_shape

    if len(axis) == 1:
        return F.interpolate(x, size=new_size[0], mode="linear", align_corners=True)
    if len(axis) == 2:
        return F.interpolate(x, size=new_size, mode="bicubic", align_corners=True)

    X = paddle.fft.rfftn(x.astype("float32"), norm="forward", axes=axis)

    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1  # Redundant last coefficient
    new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, X.shape[-len(axis) :])]
    out_fft = paddle.zeros(
        [x.shape[0], x.shape[1], *new_fft_size], dtype=paddle.complex64
    )

    mode_indexing = [((None, m // 2), (-m // 2, None)) for m in new_fft_size_c[:-1]] + [
        ((None, new_fft_size_c[-1]),)
    ]
    for i, boundaries in enumerate(itertools.product(*mode_indexing)):

        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

        out_fft[idx_tuple] = X[idx_tuple]
    y = paddle.fft.irfftn(out_fft, s=new_size, norm="forward", axes=axis)

    return y


class FactorizedTensor(nn.Layer):
    def __init__(self, shape, init_scale):
        super().__init__()
        self.shape = shape
        self.init_scale = init_scale
        self.real = self.create_parameter(
            shape=shape,
        )
        self.real = initializer.normal_(self.real, 0, init_scale)
        self.imag = self.create_parameter(shape=shape)
        self.imag = initializer.normal_(self.imag, 0, init_scale)

    def __repr__(self):
        return f"FactorizedTensor(shape={self.shape})"

    @property
    def data(self):
        return paddle.complex(self.real, self.imag)


class FactorizedSpectralConv(nn.Layer):
    """Generic N-Dimensional Fourier Neural Operator

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_modes (Tuple[int, ...]): Number of modes to use for contraction in Fourier domain during training.
            .. warning::
            We take care of the redundancy in the Fourier modes, therefore, for an input
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes
            as the real FFT is redundant along that last dimension.

            .. note::
                Provided modes should be even integers. odd numbers will be rounded to the closest even number.
                This can be updated dynamically during training.
        max_n_modes (int, optional): * If not None, **maximum** number of modes to keep
            in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
            * If None, all the n_modes are used. Defaults to None.
        bias (bool, optional): Whether to use bias in the layers. Defaults to True.
        n_layers (int, optional): Number of Fourier Layers. Defaults to 1.
        separable (bool, optional): Whether to use separable Fourier Conv. Defaults to False.
        output_scaling_factor (Optional[Union[Number, List[Number]]], optional): Scaling factor for the
            output. Defaults to None.
        rank (float, optional): Rank of the tensor factorization of the Fourier weights. Defaults to 0.5.
        factorization (str, optional): Tensor factorization of the parameters weight to use.
            * If None, a dense tensor parametrizes the Spectral convolutions
            * Otherwise, the specified tensor factorization is used. Defaults to None.
        implementation (str, optional): If factorization is not None, forward mode to use.
            * `reconstructed` : the full weight tensor is reconstructed from the
            factorization and used for the forward pass
            * `factorized` : the input is directly contracted with the factors of
            the decomposition. Defaults to "reconstructed".
        joint_factorization (bool, optional): Whether all the Fourier Layers should be parametrized by a
            single tensor. Defaults to False.
        init_std (str, optional): The std to use for the init. Defaults to "auto".
        fft_norm (str, optional):The normalization mode for the FFT. Defaults to "backward".
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
        factorization: str = None,
        implementation: str = "reconstructed",
        joint_factorization: bool = False,
        init_std: str = "auto",
        fft_norm: str = "backward",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization
        self.n_modes = n_modes

        self.order = len(self.n_modes)
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
        ] = validate_scaling_factor(output_scaling_factor, self.order, n_layers)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std
        self.fft_norm = fft_norm
        if factorization is None:
            factorization = "Dense"
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"
        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    f"To use separable Fourier Conv, in_channels must be equal to out_channels, but got in_channels={in_channels} and out_channels={out_channels}"
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
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
                    FactorizedTensor(weight_shape, init_scale=init_std)
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

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.output_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.output_scaling_factor[layer_index])
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape
        if in_shape == out_shape:
            return x
        else:
            return resample(
                x,
                1.0,
                list(range(2, x.ndim)),
                output_shape=out_shape,
            )

    def forward(
        self,
        x: paddle.Tensor,
        indices: int = 0,
        output_shape: Optional[Tuple[int]] = None,
    ):
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        x = paddle.fft.rfftn(x=x, norm=self.fft_norm, axes=fft_dims)

        if self.order > 1:
            x = paddle.fft.fftshift(x=x, axes=fft_dims[:-1])

        out_fft = paddle.zeros(
            shape=[batchsize, self.out_channels, *fft_size], dtype=paddle.complex64
        )

        starts = [
            (max_modes - min(size, n_mode))
            for size, n_mode, max_modes in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        slices_w = [slice(None), slice(None)]
        slices_w += [
            (slice(start // 2, -start // 2) if start else slice(start, None))
            for start in starts[:-1]
        ]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        w_real = self.weight[indices].real[
            slices_w[0], slices_w[1], slices_w[2], slices_w[3]
        ]
        w_imag = self.weight[indices].imag[
            slices_w[0], slices_w[1], slices_w[2], slices_w[3]
        ]

        starts = [
            (size - min(size, n_mode))
            for (size, n_mode) in zip(list(x.shape[2:]), list(w_real.shape[2:]))
        ]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        slices_x += [
            slice(start // 2, -start // 2) if start else slice(start, None)
            for start in starts[:-1]
        ]
        slices_x += [
            slice(None, -starts[-1]) if starts[-1] else slice(None)
        ]  # The last mode already has redundant half removed
        idx_tuple = slices_x
        if len(idx_tuple) == 4:
            out_fft[
                idx_tuple[0], idx_tuple[1], idx_tuple[2], idx_tuple[3]
            ] = self._contract(
                x[idx_tuple[0], idx_tuple[1], idx_tuple[2], idx_tuple[3]],
                w_real,
                w_imag,
                separable=self.separable,
            )
        elif len(idx_tuple) == 3:
            out_fft[idx_tuple[0], idx_tuple[1], idx_tuple[2]] = self._contract(
                x[idx_tuple[0], idx_tuple[1], idx_tuple[2]],
                w_real,
                w_imag,
                separable=self.separable,
            )
        else:
            raise ValueError("Not implemented")

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])
                ]
            )

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = paddle.fft.fftshift(x=out_fft, axes=fft_dims[:-1])

        x = paddle.fft.irfftn(
            x=out_fft, s=mode_sizes, axes=fft_dims, norm=self.fft_norm
        )
        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x


class FactorizedSpectralConv1d(FactorizedSpectralConv):
    """1D Spectral Conv

    This is provided for reference only,
    see :class:`FactorizedSpectralConv` for the preferred, general implementation
    """

    def forward(self, x, indices=0):
        batchsize, channels, width = x.shape

        x = paddle.fft.rfft(x, norm=self.fft_norm)

        out_fft = paddle.zeros(
            shape=[batchsize, self.out_channels, width // 2 + 1], dtype=paddle.complex64
        )

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None, self.n_modes[0]),  # :half_n_modes[0]]
        )

        w_real = self.weight[indices].real[slices[0], slices[1], slices[2]]
        w_imag = self.weight[indices].imag[slices[0], slices[1], slices[2]]

        out_fft[slices[0], slices[1], slices[2]] = self._contract(
            x[slices[0], slices[1], slices[2]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])

        x = paddle.fft.irfft(out_fft, n=width, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FactorizedSpectralConv2d(FactorizedSpectralConv):
    """2D Spectral Conv.

    This is provided for reference only,
    see :class:`FactorizedSpectralConv` for the preferred, general implementation
    """

    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x = paddle.fft.rfft2(x.float(), norm=self.fft_norm, axes=(-2, -1))

        # The output will be of size (batch_size, self.out_channels,
        # x.size(-2), x.size(-1)//2 + 1)
        out_fft = paddle.zeros(
            shape=[batchsize, self.out_channels, height, width // 2 + 1],
            dtype=paddle.complex64,
        )

        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.n_modes[0] // 2),  # :half_n_modes[0],
            slice(self.n_modes[1]),  #      :half_n_modes[1]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0]:,
            slice(self.n_modes[1]),  # ......      :half_n_modes[1]]
        )
        logger.message(
            f"2D: {x[slices0].shape=}, {self._get_weight(indices)[slices0].shape=}, {self._get_weight(indices).shape=}"
        )

        w_real = self.weight[indices].real[
            slices1[0], slices1[1], slices1[2], slices1[3]
        ]
        w_imag = self.weight[indices].imag[
            slices1[0], slices1[1], slices1[2], slices1[3]
        ]

        """Upper block (truncate high frequencies)."""
        out_fft[slices0[0], slices0[1], slices0[2], slices0[3]] = self._contract(
            x[slices0[0], slices0[1], slices0[2], slices0[3]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        w_real = self.weight[indices].real[
            slices0[0], slices0[1], slices0[2], slices0[3]
        ]
        w_imag = self.weight[indices].imag[
            slices0[0], slices0[1], slices0[2], slices0[3]
        ]

        """Lower block"""
        out_fft[slices1[0], slices1[1], slices1[2], slices1[3]] = self._contract(
            x[slices1[0], slices1[1], slices1[2], slices1[3]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[indices][0])
            height = round(height * self.output_scaling_factor[indices][1])

        x = paddle.fft.irfft2(
            out_fft, s=(height, width), axes=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class FactorizedSpectralConv3d(FactorizedSpectralConv):
    """3D Spectral Conv.

    This is provided for reference only,
    see :class:`FactorizedSpectralConv` for the preferred, general implementation
    """

    def forward(self, x, indices=0):
        batchsize, channels, height, width, depth = x.shape

        x = paddle.fft.rfftn(x.float(), norm=self.fft_norm, axes=[-3, -2, -1])

        out_fft = paddle.zeros(
            shape=[batchsize, self.out_channels, height, width, depth // 2 + 1],
            dtype=paddle.complex64,
        )

        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.n_modes[0] // 2),  # :half_n_modes[0],
            slice(self.n_modes[1] // 2),  # :half_n_modes[1],
            slice(self.n_modes[2]),  # :half_n_modes[2]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(self.n_modes[0] // 2),  # ...... :half_n_modes[0],
            slice(-self.n_modes[1] // 2, None),  # -half_n_modes[1]:,
            slice(self.n_modes[2]),  # ......      :half_n_modes[0]]
        )
        slices2 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0]:,
            slice(self.n_modes[1] // 2),  # ...... :half_n_modes[1],
            slice(self.n_modes[2]),  # ......      :half_n_modes[2]]
        )
        slices3 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0],
            slice(-self.n_modes[1] // 2, None),  # -half_n_modes[1],
            slice(self.n_modes[2]),  # ......      :half_n_modes[2]]
        )

        w_real = self.weight[indices].real[
            slices3[0], slices3[1], slices3[2], slices3[3], slices3[4]
        ]
        w_imag = self.weight[indices].imag[
            slices3[0], slices3[1], slices3[2], slices3[3], slices3[4]
        ]

        """Upper block -- truncate high frequencies."""
        out_fft[
            slices0[0], slices0[1], slices0[2], slices0[3], slices0[4]
        ] = self._contract(
            x[slices0[0], slices0[1], slices0[2], slices0[3], slices0[4]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        w_real = self.weight[indices].real[
            slices2[0], slices2[1], slices2[2], slices2[3], slices2[4]
        ]
        w_imag = self.weight[indices].imag[
            slices2[0], slices2[1], slices2[2], slices2[3], slices2[4]
        ]
        """Low-pass filter for indices 2 & 4, and high-pass filter for index 3."""
        out_fft[
            slices1[0], slices1[1], slices1[2], slices1[3], slices1[4]
        ] = self._contract(
            x[slices1[0], slices1[1], slices1[2], slices1[3], slices1[4]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        w_real = self.weight[indices].real[
            slices1[0], slices1[1], slices1[2], slices1[3], slices1[4]
        ]
        w_imag = self.weight[indices].imag[
            slices1[0], slices1[1], slices1[2], slices1[3], slices1[4]
        ]
        """Low-pass filter for indices 3 & 4, and high-pass filter for index 2."""
        out_fft[
            slices2[0], slices2[1], slices2[2], slices2[3], slices2[4]
        ] = self._contract(
            x[slices2[0], slices2[1], slices2[2], slices2[3], slices2[4]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        w_real = self.weight[indices].real[
            slices0[0], slices0[1], slices0[2], slices0[3], slices0[4]
        ]
        w_imag = self.weight[indices].imag[
            slices0[0], slices0[1], slices0[2], slices0[3], slices0[4]
        ]
        """Lower block -- low-cut filter in indices 2 & 3
        and high-cut filter in index 4."""
        out_fft[
            slices3[0], slices3[1], slices3[2], slices3[3], slices3[4]
        ] = self._contract(
            x[slices3[0], slices3[1], slices3[2], slices3[3], slices3[4]],
            w_real,
            w_imag,
            separable=self.separable,
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])
            height = round(height * self.output_scaling_factor[1])
            depth = round(depth * self.output_scaling_factor[2])

        x = paddle.fft.irfftn(
            out_fft, s=(height, width, depth), axes=[-3, -2, -1], norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x


class FNOBlocks(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers: int = 1,
        max_n_modes: int = None,
        use_mlp: bool = False,
        mlp: Optional[Dict[str, float]] = None,
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
        SpectralConv: FactorizedSpectralConv = FactorizedSpectralConv,
        joint_factorization: bool = False,
        implementation: str = "factorized",
        fft_norm: str = "forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.max_n_modes = max_n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fno_skip = fno_skip
        self.mlp_skip = mlp_skip
        self.use_mlp = use_mlp
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.stabilizer = stabilizer
        self.norm = norm

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            max_n_modes=max_n_modes,
            rank=rank,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.LayerList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if use_mlp:
            self.mlp = nn.LayerList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        hidden_channels=int(
                            round(self.out_channels * mlp["expansion"])
                        ),
                        dropout=mlp["dropout"],
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp_skips = nn.LayerList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        type=mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.LayerList(
                [
                    getattr(nn, f"InstanceNorm{self.n_dim}d")(
                        num_features=self.out_channels
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "group_norm":
            self.norm = nn.LayerList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "ada_in":
            self.norm = nn.LayerList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got {norm} but expected None or one of [instance_norm, group_norm, layer_norm]"
            )

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape=output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape=output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs.transform(x_skip_fno, index, output_shape=output_shape)
        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs.transform(
                x_skip_mlp, index, output_shape=output_shape
            )
        if self.stabilizer == "tanh":
            x = paddle.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs.transform(x_skip_fno, index, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs.transform(
                x_skip_mlp, index, output_shape=output_shape
            )

        if self.stabilizer == "tanh":
            x = paddle.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if self.mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.mlp[index](x) + x_skip_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

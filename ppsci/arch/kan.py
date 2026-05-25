# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.arch.mlp import RandomWeightFactorization
from ppsci.arch.mlp import WeightNormLinear
from ppsci.utils import initializer

"""
This is the paddle implementation of Korogonov-Arnold-Network (KAN)
the bspline implementation is based on the torch implementation [efficient-kan] by Blealtan and akkashdash
please refer to their work (https://github.com/Blealtan/efficient-kan)
we also provide the fourier base, laplacian base, legendre polynominals implementation version of KAN, which are more efficient than the original KAN
Authors: guhaohao0991(guhaohao@baidu.com)
Date:    2025/04/
"""


class KANLinear(paddle.nn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: Callable[[paddle.Tensor], paddle.Tensor] = paddle.nn.Silu,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                paddle.arange(start=-spline_order, end=grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(shape=[in_features, -1])
            .contiguous()
        )
        self.register_buffer(name="grid", tensor=grid)

        self.base_weight = self.create_parameter(
            shape=[out_features, in_features],
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(shape=[out_features, in_features])
            ),
        )
        self.spline_weight = self.create_parameter(
            shape=[out_features, in_features, grid_size + spline_order],
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(
                    shape=[out_features, in_features, grid_size + spline_order]
                )
            ),
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = self.create_parameter(
                shape=[out_features, in_features],
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.empty(shape=[out_features, in_features])
                ),
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        self.base_weight = initializer.kaiming_uniform_(
            tensor=self.base_weight,
            a=math.sqrt(5) * self.scale_base,
            nonlinearity="leaky_relu",
        )
        with paddle.no_grad():
            noise = (
                (
                    paddle.rand(
                        shape=[self.grid_size + 1, self.in_features, self.out_features]
                    )
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )

            paddle.assign(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                ),
                output=self.spline_weight.data,
            )

            if self.enable_standalone_scale_spline:
                self.spline_scaler = initializer.kaiming_uniform_(
                    tensor=self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline,
                    nonlinearity="leaky_relu",
                )

    def b_splines(self, x: paddle.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            paddle.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.shape[1] == self.in_features
        grid: paddle.Tensor = self.grid
        x = x.unsqueeze(axis=-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (
                grid[:, k:-1] - grid[:, : -(k + 1)]
            ) * bases[:, :, :-1] + (grid[:, k + 1 :] - x) / (
                grid[:, k + 1 :] - grid[:, 1:-k]
            ) * bases[
                :, :, 1:
            ]

        assert tuple(bases.shape) == (
            x.shape[0],
            self.in_features,
            self.grid_size + self.spline_order,
        )

        return bases.contiguous()

    def curve2coeff(self, x: paddle.Tensor, y: paddle.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).
            y (paddle.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            paddle.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.shape[1] == self.in_features
        assert tuple(y.shape) == (x.shape[0], self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            perm=dim2perm(self.b_splines(x).ndim, 0, 1)
        )  # [in_features, batch_size, grid_size + spline_order]
        B = y.transpose(
            perm=dim2perm(y.ndim, 0, 1)
        )  # [in_features, batch_size, out_features]
        solution = paddle.linalg.lstsq(x=A, y=B)[
            0
        ]  # [in_features, grid_size + spline_order, out_features]
        if A.shape[0] == 1:
            solution = solution.unsqueeze(axis=0)
        # print("A shape: ", A.shape, "B shape: ", B.shape, "Solution shape: ", solution.shape)
        result = solution.transpose([2, 0, 1])
        assert tuple(result.shape) == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(axis=-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: paddle.Tensor):
        assert x.dim() == 2 and x.shape[1] == self.in_features

        base_output = paddle.nn.functional.linear(
            x=self.base_activation(x), weight=self.base_weight.T
        )

        spline_output = paddle.nn.functional.linear(
            x=self.b_splines(x).reshape([x.shape[0], -1]).contiguous(),
            weight=self.scaled_spline_weight.reshape(
                [self.out_features, -1]
            ).T.contiguous(),
        )
        # cant calculate 1st order derivation using view
        # spline_output = paddle.nn.functional.linear(
        #     x=self.b_splines(x).view(x.shape[0], -1),
        #     weight=self.scaled_spline_weight.view(self.out_features, -1).T)

        return base_output + spline_output

    @paddle.no_grad()
    def update_grid(self, x: paddle.Tensor, margin=0.01):
        assert x.dim() == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]

        splines = self.b_splines(x)  # [batch, in, coeff]
        splines = splines.transpose(perm=[1, 0, 2])  # [in, batch, coeff]
        orig_coeff = self.scaled_spline_weight  # [out, in, coeff]
        orig_coeff = orig_coeff.transpose(perm=[1, 2, 0])  # [in, coeff, out]
        unreduced_spline_output = paddle.bmm(
            x=splines, y=orig_coeff
        )  # [in, batch, out]
        unreduced_spline_output = unreduced_spline_output.transpose(
            perm=[1, 0, 2]
        )  # [batch, in, out]

        # sort each channel individually to collect data distribution
        x_sorted = (paddle.sort(x=x, axis=0), paddle.argsort(x=x, axis=0))[0]
        grid_adaptive = x_sorted[
            paddle.linspace(
                start=0, stop=batch - 1, num=self.grid_size + 1, dtype="int64"
            )
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            paddle.arange(
                dtype=paddle.get_default_dtype(), end=self.grid_size + 1
            ).unsqueeze(axis=1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = paddle.concat(
            x=[
                grid[:1]
                - uniform_step
                * paddle.arange(
                    start=self.spline_order,
                    end=0,
                    step=-1,
                    dtype=paddle.get_default_dtype(),
                ).unsqueeze(axis=1),
                grid,
                grid[-1:]
                + uniform_step
                * paddle.arange(
                    start=1, end=self.spline_order + 1, dtype=paddle.get_default_dtype()
                ).unsqueeze(axis=1),
            ],
            axis=0,
        )

        paddle.assign(grid.T, output=self.grid)
        paddle.assign(
            self.curve2coeff(x, unreduced_spline_output), output=self.spline_weight.data
        )

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        """
        l1_fake = self.spline_weight.abs().mean(axis=-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -paddle.sum(x=p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(base.Arch):
    """Kolmogorov-Arnold Network (KAN).

    Args:
        layers_hidden (Tuple[int, ...]): The number of hidden neurons in each layer.
        input_keys (Tuple[str, ...]): The keys of the input dictionary.
        output_keys (Tuple[str, ...]): The keys of the output dictionary.
        grid_size (int): The size of the grid used by the spline basis functions. Default: 5.
        spline_order (int): The order of the spline basis functions. Default: 3.
        scale_noise (float): The scaling factor for the noise added to the weights of the KAN-linear layers. Default: 0.1.
        scale_base (float): The scaling factor for the base activation output. Default: 1.0.
        scale_spline (float): The scaling factor for the b-spline output. Default: 1.0.
        base_activation (Callable[[paddle.Tensor], paddle.Tensor]): The base activation function. Default: paddle.nn.Silu.
        grid_eps (float): The epsilon value used to initialize the grid. Default: 0.02.
        grid_range (Tuple[float, float]): The domain range of the grid for b-spline interpolation. Default: (-1, 1).

    Examples:
        >>> import paddle
        >>> import ppsci
        >>> model = ppsci.arch.KAN(
        ...     layers_hidden=(2, 5, 5, 1),
        ...     input_keys=("x", "y"),
        ...     output_keys=("z"),
        ...     grid_size=5,
        ...     spline_order=3
        >>> )
        >>> input_dict = {"x": paddle.rand([64, 1]),
        ...               "y": paddle.rand([64, 1])}
        >>> output_dict = model(input_dict)
        >>> print(output_dict["z"].shape)
        [64, 1]
    """

    def __init__(
        self,
        layers_hidden: Tuple[int, ...],
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: Callable[[paddle.Tensor], paddle.Tensor] = paddle.nn.Silu,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1, 1),
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = paddle.nn.LayerList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x_dict, update_grid=False):
        x = self.concat_to_tensor(x_dict, self.input_keys, axis=-1)
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers) - 1:
                x = paddle.nn.functional.tanh(x=x)
        out_dic = self.split_to_dict(x, self.output_keys, axis=-1)
        return out_dic

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class KAN_Legendre(paddle.nn.Layer):
    def __init__(self, input_features, output_features, max_degree):

        super(KAN_Legendre, self).__init__()
        self.max_degree = max_degree
        self.input_features = input_features
        self.output_features = output_features

        self.poly_weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(
                shape=[max_degree + 1, self.input_features, self.output_features]
            )
        )  # Legendre polynomicals weight matrix, coefficients of each polynominal terms
        init_Orthogonal = nn.initializer.Orthogonal()
        init_Orthogonal(
            self.poly_weights
        )  # Legendre polynominals satisfy Orthogonality and Completeness

        self.dropout = nn.Dropout(p=0.1)
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=self.output_features)
        )

    def forward(self, x):
        batch_size = tuple(x.shape)[0]
        # Legendre polynominals are calculated by the Rodrigues' formula,
        # given P0 and P1, any Pm can be recurrently calculated according to the ortogonal and complete conditions.
        P_n_minus_2 = paddle.ones(shape=(batch_size, self.input_features))  # P_0
        P_n_minus_1 = x.clone()
        polys = [
            P_n_minus_2.unsqueeze(axis=-1),
            P_n_minus_1.unsqueeze(axis=-1),
        ]  # lists of polynominals

        for n in range(2, self.max_degree + 1):
            P_n = ((2 * n - 1) * x * P_n_minus_1 - (n - 1) * P_n_minus_2) / n
            polys.append(P_n.unsqueeze(axis=-1))
            P_n_minus_2 = P_n_minus_1
            P_n_minus_1 = P_n
        polys = paddle.concat(
            x=polys, axis=-1
        )  # polynominal tensor, shape [batch_size, input_features, max_degree]
        polys = self.dropout(polys)
        output = paddle.einsum("bif, fio->bo", polys, self.poly_weights) + self.bias
        return output


class KAN_Laplace(paddle.nn.Layer):
    def __init__(
        self, input_features, output_features, initial_grid_size, addbias=True
    ):
        super(KAN_Laplace, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.addbias = addbias
        self.grid_size_param = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(
                data=initial_grid_size, dtype=paddle.get_default_dtype()
            )
        )
        self.laplace_coeffs = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(
                shape=[2, output_features, input_features, initial_grid_size]
            )
        )
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.laplace_coeffs)
        if self.addbias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[1, output_features])
            )

    def forward(self, x):
        grid_size = paddle.clip(x=self.grid_size_param, min=1).round().astype("int32")
        xshape = tuple(x.shape)
        outshape = xshape[:-1] + (self.output_features,)
        x = paddle.reshape(x=x, shape=(-1, self.input_features))
        lambdas = paddle.reshape(
            x=paddle.linspace(start=0.1, stop=1.0, num=grid_size),
            shape=(1, 1, 1, grid_size),
        )
        x_rshape = paddle.reshape(
            x=x, shape=(tuple(x.shape)[0], 1, tuple(x.shape)[1], 1)
        )
        exp_neg = paddle.exp(x=-lambdas * x_rshape)
        exp_pos = paddle.exp(x=lambdas * x_rshape)
        y = paddle.sum(
            x=exp_neg * self.laplace_coeffs[0:1, :, :, :grid_size], axis=(-2, -1)
        )
        y += paddle.sum(
            x=exp_pos * self.laplace_coeffs[0:1, :, :, :grid_size], axis=(-2, -1)
        )
        if self.addbias:
            y += self.bias

        y = paddle.reshape(x=y, shape=outshape)
        return y


class KAN_Fourier(paddle.nn.Layer):
    def __init__(
        self, input_features, output_features, initial_grid_size, addbias=True
    ):
        super(KAN_Fourier, self).__init__()
        self.addbias = addbias
        self.input_features = input_features
        self.output_features = output_features
        self.grid_size_param = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(
                data=initial_grid_size, dtype=paddle.get_default_dtype()
            )
        )
        self.fourier_coeffs = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(
                shape=[2, output_features, input_features, initial_grid_size]
            )
        )
        # init_XavierUniform = nn.initializer.XavierUniform()
        # init_XavierUniform(self.fourier_coeffs)
        # initializer.glorot_normal_(self.fourier_coeffs)
        initializer.xavier_uniform_(self.fourier_coeffs)
        if self.addbias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(
                    shape=[1, output_features], dtype=paddle.get_default_dtype()
                )
            )

    def forward(self, x):
        # x = paddle.cast(x, "float32")
        out_shape = tuple(x.shape)[:-1] + (self.output_features,)

        grid_size = (
            paddle.clip(x=self.grid_size_param, min=1).round().astype(dtype="int32")
        )
        k = paddle.arange(
            start=1, end=grid_size + 1, dtype=paddle.get_default_dtype()
        ).reshape((1, 1, 1, -1))

        x_rshape = x.reshape((-1, 1, x.shape[-1], 1))

        c = paddle.cos(x=k * x_rshape)
        s = paddle.sin(x=k * x_rshape)

        # coeffs = paddle.cast(self.fourier_coeffs[:, :, :, :grid_size], "float32")
        coeffs = self.fourier_coeffs[:, :, :, :grid_size]

        y = paddle.sum(x=c * coeffs[0:1], axis=(-2, -1)) + paddle.sum(
            x=s * coeffs[1:2], axis=(-2, -1)
        )

        if self.addbias:
            # y += paddle.cast(self.bias, "float32")
            y += self.bias

        y = paddle.reshape(x=y, shape=out_shape)

        return y


class RandomWeightKanFactorization(nn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_grid_size: int = 10,
        bias: bool = True,
        mean: float = 0.5,
        std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kan_layer = KAN_Fourier(
            input_features=in_features,
            output_features=out_features,
            initial_grid_size=init_grid_size,
            addbias=bias,
        )
        self.weight_g = self.create_parameter((out_features,))
        # self.__init_weights(mean, std)

    def _init_weights(self, mean, std):
        with paddle.no_grad():
            nn.initializer.Normal(mean, std)(self.weight_g)
            paddle.assign(paddle.exp(self.weight_g), self.weight_g)

        self.weight_g.stop_gradient = False
        self.kan_layer.fourier_coeffs.stop_gradient = False
        # self.kan_layer.grid_size_param.stop_gradient = False  # 需要测试是否需要训练，效果不知道
        if hasattr(self.kan_layer, "bias") and self.kan_layer.bias is not None:
            self.kan_layer.bias.stop_gradient = False

    def forward(self, input):
        return self.kan_layer(input) * self.weight_g.reshape((1, -1))


class PeriodEmbedding(nn.Layer):
    def __init__(self, periods: Dict[str, Tuple[float, bool]]):
        super().__init__()
        self.freqs_dict = {
            k: self.create_parameter(
                [],
                attr=paddle.ParamAttr(trainable=trainable),
                default_initializer=nn.initializer.Constant(2 * np.pi / float(p)),
            )  # mu = 2*pi / period for sin/cos function
            for k, (p, trainable) in periods.items()
        }
        self.freqs = nn.ParameterList(list(self.freqs_dict.values()))

    def forward(self, x: Dict[str, paddle.Tensor]):
        y = {k: v for k, v in x.items()}  # shallow copy to avoid modifying input dict

        for k, w in self.freqs_dict.items():
            y[k] = paddle.concat([paddle.cos(w * x[k]), paddle.sin(w * x[k])], axis=-1)

        return y


class FourierEmbedding(nn.Layer):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, but got {out_features}.")

        self.kernel = self.create_parameter(
            [in_features, out_features // 2],
            default_initializer=nn.initializer.Normal(std=scale),
        )

    def forward(self, x: paddle.Tensor):
        y = paddle.concat(
            [
                paddle.cos(x @ self.kernel),
                paddle.sin(x @ self.kernel),
            ],
            axis=-1,
        )
        return y


class PiraKanBlock(nn.Layer):
    def __init__(
        self,
        embed_dim: int,
        kan_grid_size: int = 10,
        activation: str = "tanh",
        random_weight: Optional[Dict[str, float]] = None,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.kan_layer1 = (
            KAN_Fourier(embed_dim, embed_dim, kan_grid_size)
            if random_weight is None
            else RandomWeightKanFactorization(
                embed_dim,
                embed_dim,
                kan_grid_size,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        )
        self.kan_layer2 = (
            KAN_Fourier(embed_dim, embed_dim, kan_grid_size)
            if random_weight is None
            else RandomWeightKanFactorization(
                embed_dim,
                embed_dim,
                kan_grid_size,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        )
        self.kan_layer3 = (
            KAN_Fourier(embed_dim, embed_dim, kan_grid_size)
            if random_weight is None
            else RandomWeightKanFactorization(
                embed_dim,
                embed_dim,
                kan_grid_size,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        )
        self.alpha = self.create_parameter(
            [
                1,
            ],
            default_initializer=nn.initializer.Constant(alpha_init),
        )
        self.act1 = (
            act_mod.get_activation(activation)
            if activation != "stan"
            else act_mod.get_activation(activation)(embed_dim)
        )
        self.act2 = (
            act_mod.get_activation(activation)
            if activation != "stan"
            else act_mod.get_activation(activation)(embed_dim)
        )
        self.act3 = (
            act_mod.get_activation(activation)
            if activation != "stan"
            else act_mod.get_activation(activation)(embed_dim)
        )

    def forward(self, x, u, v):
        f = self.act1(self.kan_layer1(x))
        z1 = f * u + (1 - f) * v
        g = self.act2(self.kan_layer2(z1))
        z2 = g * u + (1 - g) * v
        h = self.act3(self.kan_layer3(z2))
        out = self.alpha * h + (1 - self.alpha) * x
        return out


class PiraKanNet(base.Arch):
    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_blocks: int,
        hidden_size: int,
        activation: str = "tanh",
        weight_norm: bool = False,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        kan_grid_size: int = 10,
        periods: Optional[Dict[int, Tuple[float, bool]]] = None,
        fourier: Optional[Dict[str, Union[float, int]]] = None,
        random_weight: Optional[Dict[str, float]] = None,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.blocks = []
        self.periods = periods
        self.fourier = fourier
        if periods:
            self.period_emb = PeriodEmbedding(periods)

        if isinstance(hidden_size, int):
            if not isinstance(num_blocks, int):
                raise ValueError("num_blocks should be an int")
            hidden_size = [hidden_size] * num_blocks
        else:
            raise ValueError(f"hidden_size should be int, but got {type(hidden_size)}")

        # initialize FC layer(s)
        cur_size = len(self.input_keys) if input_dim is None else input_dim
        if input_dim is None and periods:
            # period embeded channel(s) will be doubled automatically
            # if input_dim is not specified
            cur_size += len(periods)

        if fourier:
            self.fourier_emb = FourierEmbedding(
                cur_size, fourier["dim"], fourier["scale"]
            )
            cur_size = fourier["dim"]

        self.embed_u = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else (
                    nn.Linear(cur_size, hidden_size[0])
                    if random_weight is None
                    else RandomWeightFactorization(
                        cur_size,
                        hidden_size[0],
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            ),
            (
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(hidden_size[0])
            ),
        )
        self.embed_v = nn.Sequential(
            (
                WeightNormLinear(cur_size, hidden_size[0])
                if weight_norm
                else (
                    nn.Linear(cur_size, hidden_size[0])
                    if random_weight is None
                    else RandomWeightFactorization(
                        cur_size,
                        hidden_size[0],
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                )
            ),
            (
                act_mod.get_activation(activation)
                if activation != "stan"
                else act_mod.get_activation(activation)(hidden_size[0])
            ),
        )

        for i, _size in enumerate(hidden_size):
            self.blocks.append(
                PiraKanBlock(
                    embed_dim=cur_size,
                    kan_grid_size=kan_grid_size,
                    activation=activation,
                    random_weight=random_weight,
                    alpha_init=alpha_init,
                )
            )
            cur_size = _size

        self.blocks = nn.LayerList(self.blocks)
        if random_weight:
            self.last_fc = RandomWeightFactorization(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
                mean=random_weight["mean"],
                std=random_weight["std"],
            )
        else:
            self.last_fc = nn.Linear(
                cur_size,
                len(self.output_keys) if output_dim is None else output_dim,
            )

    def forward_tensor(self, x):
        u = self.embed_u(x)
        v = self.embed_v(x)

        y = x
        for i, block in enumerate(self.blocks):
            y = block(y, u, v)

        y = self.last_fc(y)
        return y

    def forward(self, x):
        if self._input_transform is not None:
            x_ = self._input_transform(x)
        else:
            x_ = x

        if self.periods:
            x_ = self.period_emb(x_)

        y = self.concat_to_tensor(x_, self.input_keys, axis=-1)

        if self.fourier:
            y = self.fourier_emb(y)

        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y

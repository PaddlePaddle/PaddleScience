from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppsci.arch import base
from ppsci.arch import fno_block


class UNONet(base.Arch):
    """N-Dimensional U-Shaped Neural Operator

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        n_modes (Tuple[int, ...]): number of modes to keep in Fourier Layer, along each dimension
            The dimensionality of the TFNO is inferred from ``len(n_modes)`
        hidden_channels (int): Width of the FNO (i.e. number of channels)
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        lifting_channels (int, optional): Number of hidden channels of the lifting block of the FNO.
            Defaults to 256.
        projection_channels (int, optional): Number of hidden channels of the projection block of the FNO.
            Defaults to 256.
        n_layers (int, optional): Number of Fourier Layers. Defaults to 4.
        use_mlp (bool, optional): Whether to use an MLP layer after each FNO block. Defaults to False.
        mlp (dict[str, float], optional): Parameters of the MLP. {'expansion': float, 'dropout': float}.
            Defaults to None.
        non_linearity (nn.Layer, optional): Non-Linearity module to use. Defaults to F.gelu.
        norm (F.module, optional): Normalization layer to use. Defaults to None.
        preactivation (bool, optional): Whether to use resnet-style preactivation. Defaults to False.
        skip (str, optional): Type of skip connection to use,{'linear', 'identity', 'soft-gating'}.
            Defaults to "soft-gating".
        separable (bool, optional): Whether to use a depthwise separable spectral convolution.
            Defaults to  False.
        factorization (str, optional): Tensor factorization of the parameters weight to use.
            * If None, a dense tensor parametrizes the Spectral convolutions.
            * Otherwise, the specified tensor factorization is used. Defaults to "Tucker".
        rank (float, optional): Rank of the tensor factorization of the Fourier weights. Defaults to 1.0.
        joint_factorization (bool, optional): Whether all the Fourier Layers should be parametrized by a
            single tensor (vs one per layer). Defaults to False.
        fixed_rank_modes (bool, optional): Modes to not factorize. Defaults to False.
        implementation (str, optional): {'factorized', 'reconstructed'}, optional. Defaults to "factorized".
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass.
            * `factorized` : the input is directly contracted with the factors of the decomposition.
        decomposition_kwargs (dict, optional): Optionaly additional parameters to pass to the tensor
            decomposition. Defaults to dict().
        domain_padding (str, optional): Whether to use percentage of padding. Defaults to None.
        domain_padding_mode (str, optional): {'symmetric', 'one-sided'}, optional
            How to perform domain padding, by default 'one-sided'. Defaults to "one-sided".
        fft_norm (str, optional): The normalization mode for the FFT. Defaults to "forward".
        patching_levels (int, optional): Number of patching levels to use. Defaults to 0.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        in_channels,
        out_channels,
        hidden_channels,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        uno_out_channels=None,
        uno_n_modes=None,
        uno_scalings=None,
        horizontal_skips_map=None,
        incremental_n_modes=None,
        use_mlp=False,
        mlp=None,
        non_linearity=F.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        horizontal_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        patching_levels=0,
        normalizer=None,
        **kwargs,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        if uno_out_channels is None:
            raise ValueError("uno_out_channels can not be None")
        if uno_n_modes is None:
            raise ValueError("uno_n_modes can not be None")
        if uno_scalings is None:
            raise ValueError("uno_scalings can not be None")

        if len(uno_out_channels) != n_layers:
            raise ValueError("Output channels for all layers are not given")

        if len(uno_n_modes) != n_layers:
            raise ValueError("Number of modes for all layers are not given")

        if len(uno_scalings) != n_layers:
            raise ValueError("Scaling factor for all layers are not given")

        self.n_dim = len(uno_n_modes[0])
        self.uno_out_channels = uno_out_channels
        self.uno_n_modes = uno_n_modes
        self.uno_scalings = uno_scalings

        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        if patching_levels:
            self.in_channels = self.in_channels * patching_levels + 1
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.horizontal_skips_map = horizontal_skips_map
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.fno_skip = (fno_skip,)
        self.mlp_skip = (mlp_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self._incremental_n_modes = incremental_n_modes
        self.mlp = mlp
        # constructing default skip maps
        if self.horizontal_skips_map is None:
            self.horizontal_skips_map = {}
            for i in range(
                0,
                n_layers // 2,
            ):
                # example, if n_layers = 5, then 4:0, 3:1
                self.horizontal_skips_map[n_layers - i - 1] = i
        # self.uno_scalings may be a 1d list specifying uniform scaling factor at each layer
        # or a 2d list, where each row specifies scaling factors along each dimention.
        # To get the final (end to end) scaling factors we need to multiply
        # the scaling factors (a list) of all layer.

        self.end_to_end_scaling_factor = [1] * len(self.uno_scalings[0])
        # multiplying scaling factors
        for k in self.uno_scalings:
            self.end_to_end_scaling_factor = [
                i * j for (i, j) in zip(self.end_to_end_scaling_factor, k)
            ]

        # list with a single element is replaced by the scaler.
        if len(self.end_to_end_scaling_factor) == 1:
            self.end_to_end_scaling_factor = self.end_to_end_scaling_factor[0]

        if isinstance(self.end_to_end_scaling_factor, (float, int)):
            self.end_to_end_scaling_factor = [
                self.end_to_end_scaling_factor
            ] * self.n_dim

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

        self.lifting = fno_block.MLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )

        self.fno_blocks = nn.LayerList([])
        self.horizontal_skips = nn.LayerDict({})
        prev_out = self.hidden_channels
        for i in range(self.n_layers):
            if i in self.horizontal_skips_map.keys():
                prev_out = (
                    prev_out + self.uno_out_channels[self.horizontal_skips_map[i]]
                )
            self.fno_blocks.append(
                fno_block.FNOBlocks(
                    in_channels=prev_out,
                    out_channels=self.uno_out_channels[i],
                    n_modes=self.uno_n_modes[i],
                    use_mlp=use_mlp,
                    mlp=mlp,
                    output_scaling_factor=[self.uno_scalings[i]],
                    non_linearity=non_linearity,
                    norm=norm,
                    preactivation=preactivation,
                    fno_skip=fno_skip,
                    mlp_skip=mlp_skip,
                    separable=separable,
                    incremental_n_modes=incremental_n_modes,
                    factorization=factorization,
                    rank=rank,
                    SpectralConv=fno_block.FactorizedSpectralConv,
                    joint_factorization=joint_factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    fft_norm=fft_norm,
                )
            )

            if i in self.horizontal_skips_map.values():
                self.horizontal_skips[str(i)] = fno_block.skip_connection(
                    self.uno_out_channels[i],
                    self.uno_out_channels[i],
                    type=horizontal_skip,
                    n_dim=self.n_dim,
                )
            prev_out = self.uno_out_channels[i]

        self.projection = fno_block.MLP(
            in_channels=prev_out,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, **kwargs):
        x = self.concat_to_tensor(x, self.input_keys)
        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling_factor)
        ]

        skip_outputs = {}
        cur_output = None
        for layer_idx in range(self.n_layers):
            if layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                output_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                output_scaling_factors = output_scaling_factors[-1 * self.n_dim :]
                t = fno_block.resample(
                    skip_val, output_scaling_factors, list(range(-self.n_dim, 0))
                )
                x = paddle.concat([x, t], axis=1)

            if layer_idx == self.n_layers - 1:
                cur_output = output_shape
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output)
            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        out = self.projection(x)
        return {self.output_keys[0]: out}

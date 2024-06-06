from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import paddle.nn.functional as F
from paddle import nn

from ppsci.arch import base
from ppsci.arch import fno_block


class FNONet(base.Arch):
    """N-Dimensional Tensorized Fourier Neural Operator.

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
        mlp (Dict[str, float], optional): Parameters of the MLP. {'expansion': float, 'dropout': float}.
            Defaults to None.
        non_linearity (nn.functional, optional): Non-Linearity module to use. Defaults to F.gelu.
        norm (str, optional): Normalization layer to use. Defaults to None.
        ada_in_features (int,optional): The input channles of the adaptive normalization.Defaults to None.s
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
        implementation (str, optional): {'factorized', 'reconstructed'}, optional. Defaults to "factorized".
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass.
            * `factorized` : the input is directly contracted with the factors of the decomposition.
        domain_padding (Optional[Union[list,float,int]]): Whether to use percentage of padding. Defaults to
            None.
        domain_padding_mode (str, optional): {'symmetric', 'one-sided'}, optional
            How to perform domain padding, by default 'one-sided'. Defaults to "one-sided".
        fft_norm (str, optional): The normalization mode for the FFT. Defaults to "forward".
        patching_levels (int, optional): Number of patching levels to use. Defaults to 0.
        SpectralConv (nn.layer, optional): Spectral convolution layer to use.
            Defaults to fno_block.FactorizedSpectralConv.
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
        domain_padding: Optional[Union[list, float, int]] = None,
        domain_padding_mode: str = "one-sided",
        fft_norm: str = "forward",
        patching_levels: int = 0,
        SpectralConv: nn.Layer = fno_block.FactorizedSpectralConv,
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
            SpectralConv=SpectralConv,
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
        """TFNO's forward pass"""
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


class TFNO1dNet(FNONet):
    """1D Fourier Neural Operator.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        n_modes_height (Tuple[int, ...]): Number of Fourier modes to keep along the height, along each
            dimension.
        hidden_channels (int): Width of the FNO (i.e. number of channels).
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
        non_linearity (nn.functional, optional): Non-Linearity module to use. Defaults to F.gelu.
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
        implementation (str, optional): {'factorized', 'reconstructed'}, optional. Defaults to "factorized".
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass.
            * `factorized` : the input is directly contracted with the factors of the decomposition.
        domain_padding (Optional[Union[list, float, int]], optional): Whether to use percentage of padding.
            Defaults to None.
        domain_padding_mode (str, optional): {'symmetric', 'one-sided'}, optional
            How to perform domain padding, by default 'one-sided'. Defaults to "one-sided".
        fft_norm (str, optional): The normalization mode for the FFT. Defaults to "forward".
        patching_levels (int, optional): Number of patching levels to use. Defaults to 0.
        SpectralConv (nn.layer, optional): Spectral convolution layer to use.
            Defaults to fno_block.FactorizedSpectralConv.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        n_modes_height: Tuple[int, ...],
        hidden_channels: int,
        in_channels: int = 3,
        out_channels: int = 1,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        non_linearity: nn.functional = F.gelu,
        use_mlp: bool = False,
        mlp: Optional[Dict[str, float]] = None,
        norm: str = None,
        skip: str = "soft-gating",
        separable: bool = False,
        preactivation: bool = False,
        factorization: str = "Tucker",
        rank: float = 1.0,
        joint_factorization: bool = False,
        implementation: str = "factorized",
        domain_padding: Optional[Union[list, float, int]] = None,
        domain_padding_mode: str = "one-sided",
        fft_norm: str = "forward",
        patching_levels: int = 0,
        SpectralConv: nn.Layer = fno_block.FactorizedSpectralConv,
        **kwargs,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            n_modes=(n_modes_height,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=non_linearity,
            use_mlp=use_mlp,
            mlp=mlp,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            implementation=implementation,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
            patching_levels=patching_levels,
            SpectralConv=SpectralConv,
        )
        self.n_modes_height = n_modes_height


class TFNO2dNet(FNONet):
    """2D Fourier Neural Operator.

    Args:
       input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
       output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
       n_modes_height (int): number of Fourier modes to keep along the height.
       n_modes_width (int): number of modes to keep in Fourier Layer, along the width.
       hidden_channels (int): Width of the FNO (i.e. number of channels).
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
       implementation (str, optional): {'factorized', 'reconstructed'}, optional. Defaults to "factorized".
           If factorization is not None, forward mode to use::
           * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass.
           * `factorized` : the input is directly contracted with the factors of the decomposition.
       domain_padding (Union[list,float,int], optional): Whether to use percentage of padding. Defaults to
            None.
       domain_padding_mode (str, optional): {'symmetric', 'one-sided'}, optional
           How to perform domain padding, by default 'one-sided'. Defaults to "one-sided".
       fft_norm (str, optional): The normalization mode for the FFT. Defaults to "forward".
       patching_levels (int, optional): Number of patching levels to use. Defaults to 0.
       SpectralConv (nn.layer, optional): Spectral convolution layer to use.
            Defaults to fno_block.FactorizedSpectralConv.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        n_modes_height: int,
        n_modes_width: int,
        hidden_channels: int,
        in_channels: int = 3,
        out_channels: int = 1,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        non_linearity: nn.functional = F.gelu,
        use_mlp: bool = False,
        mlp: Optional[Dict[str, float]] = None,
        norm: str = None,
        skip: str = "soft-gating",
        separable: bool = False,
        preactivation: bool = False,
        factorization: str = "Tucker",
        rank: float = 1.0,
        joint_factorization: bool = False,
        implementation: str = "factorized",
        domain_padding: Optional[Union[list, float, int]] = None,
        domain_padding_mode: str = "one-sided",
        fft_norm: str = "forward",
        patching_levels: int = 0,
        SpectralConv: nn.layer = fno_block.FactorizedSpectralConv,
        **kwargs,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=non_linearity,
            use_mlp=use_mlp,
            mlp=mlp,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            implementation=implementation,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
            patching_levels=patching_levels,
            SpectralConv=SpectralConv,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width


class TFNO3dNet(FNONet):
    """3D Fourier Neural Operator.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        n_modes_height (int): Number of Fourier modes to keep along the height.
        n_modes_width (int): Number of modes to keep in Fourier Layer, along the width.
        n_modes_depth (int): Number of Fourier modes to keep along the depth.
        hidden_channels (int): Width of the FNO (i.e. number of channels).
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
        implementation (str, optional): {'factorized', 'reconstructed'}, optional. Defaults to "factorized".
            If factorization is not None, forward mode to use::
            * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass.
            * `factorized` : the input is directly contracted with the factors of the decomposition.
        domain_padding (str, optional): Whether to use percentage of padding. Defaults to None.
        domain_padding_mode (str, optional): {'symmetric', 'one-sided'}, optional
            How to perform domain padding, by default 'one-sided'. Defaults to "one-sided".
        fft_norm (str, optional): The normalization mode for the FFT. Defaults to "forward".
        patching_levels (int, optional): Number of patching levels to use. Defaults to 0.
        SpectralConv (nn.layer, optional): Spectral convolution layer to use. Defaults to fno_block.
            FactorizedSpectralConv.
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        n_modes_height: int,
        n_modes_width: int,
        n_modes_depth: int,
        hidden_channels: int,
        in_channels: int = 3,
        out_channels: int = 1,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        non_linearity: nn.functional = F.gelu,
        use_mlp: bool = False,
        mlp: Optional[Dict[str, float]] = None,
        norm: str = None,
        skip: str = "soft-gating",
        separable: bool = False,
        preactivation: bool = False,
        factorization: str = "Tucker",
        rank: float = 1.0,
        joint_factorization: bool = False,
        implementation: str = "factorized",
        domain_padding: Optional[Union[list, float, int]] = None,
        domain_padding_mode: str = "one-sided",
        fft_norm: str = "forward",
        patching_levels: int = 0,
        SpectralConv: nn.layer = fno_block.FactorizedSpectralConv,
        **kwargs,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            non_linearity=non_linearity,
            use_mlp=use_mlp,
            mlp=mlp,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            implementation=implementation,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
            patching_levels=patching_levels,
            SpectralConv=SpectralConv,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_height = n_modes_height

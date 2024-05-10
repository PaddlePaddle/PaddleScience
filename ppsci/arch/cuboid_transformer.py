from typing import Sequence
from typing import Tuple
from typing import Union

import paddle
from paddle import nn

import ppsci.arch.cuboid_transformer_decoder as cuboid_decoder
import ppsci.arch.cuboid_transformer_encoder as cuboid_encoder
import ppsci.arch.cuboid_transformer_utils as cuboid_utils
from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.arch.cuboid_transformer_encoder import NEGATIVE_SLOPE
from ppsci.utils import initializer

"""A space-time Transformer with Cuboid Attention"""


class InitialEncoder(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        out_dim,
        downsample_scale: Union[int, Sequence[int]],
        num_conv_layers: int = 2,
        activation: str = "leaky",
        padding_type: str = "nearest",
        conv_init_mode: str = "0",
        linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(InitialEncoder, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        conv_block = []
        for i in range(num_conv_layers):
            if i == 0:
                conv_block.append(
                    paddle.nn.Conv2D(
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        in_channels=dim,
                        out_channels=out_dim,
                    )
                )
                conv_block.append(
                    paddle.nn.GroupNorm(num_groups=16, num_channels=out_dim)
                )
                conv_block.append(
                    act_mod.get_activation(activation)
                    if activation != "leaky_relu"
                    else nn.LeakyReLU(NEGATIVE_SLOPE)
                )
            else:
                conv_block.append(
                    paddle.nn.Conv2D(
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        in_channels=out_dim,
                        out_channels=out_dim,
                    )
                )
                conv_block.append(
                    paddle.nn.GroupNorm(num_groups=16, num_channels=out_dim)
                )
                conv_block.append(
                    act_mod.get_activation(activation)
                    if activation != "leaky_relu"
                    else nn.LeakyReLU(NEGATIVE_SLOPE)
                )
        self.conv_block = paddle.nn.Sequential(*conv_block)
        if isinstance(downsample_scale, int):
            patch_merge_downsample = (1, downsample_scale, downsample_scale)
        elif len(downsample_scale) == 2:
            patch_merge_downsample = (1, *downsample_scale)
        elif len(downsample_scale) == 3:
            patch_merge_downsample = tuple(downsample_scale)
        else:
            raise NotImplementedError(
                f"downsample_scale {downsample_scale} format not supported!"
            )
        self.patch_merge = cuboid_encoder.PatchMerging3D(
            dim=out_dim,
            out_dim=out_dim,
            padding_type=padding_type,
            downsample=patch_merge_downsample,
            linear_init_mode=linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(
                m,
                conv_mode=self.conv_init_mode,
                linear_mode=self.linear_init_mode,
                norm_mode=self.norm_init_mode,
            )

    def forward(self, x):
        """x --> [K x Conv2D] --> PatchMerge

        Args:
            x: (B, T, H, W, C)

        Returns:
            out: (B, T, H_new, W_new, C_out)
        """

        B, T, H, W, C = x.shape

        if self.num_conv_layers > 0:
            x = x.reshape([B * T, H, W, C]).transpose(perm=[0, 3, 1, 2])
            x = self.conv_block(x).transpose(perm=[0, 2, 3, 1])
            x = self.patch_merge(x.reshape([B, T, H, W, -1]))
        else:
            x = self.patch_merge(x)
        return x


class FinalDecoder(paddle.nn.Layer):
    def __init__(
        self,
        target_thw: Tuple[int, ...],
        dim: int,
        num_conv_layers: int = 2,
        activation: str = "leaky",
        conv_init_mode: str = "0",
        linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(FinalDecoder, self).__init__()
        self.target_thw = target_thw
        self.dim = dim
        self.num_conv_layers = num_conv_layers
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        conv_block = []
        for i in range(num_conv_layers):
            conv_block.append(
                paddle.nn.Conv2D(
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    in_channels=dim,
                    out_channels=dim,
                )
            )
            conv_block.append(paddle.nn.GroupNorm(num_groups=16, num_channels=dim))
            conv_block.append(
                act_mod.get_activation(activation)
                if activation != "leaky_relu"
                else nn.LeakyReLU(NEGATIVE_SLOPE)
            )
        self.conv_block = paddle.nn.Sequential(*conv_block)
        self.upsample = cuboid_decoder.Upsample3DLayer(
            dim=dim,
            out_dim=dim,
            target_size=target_thw,
            kernel_size=3,
            conv_init_mode=conv_init_mode,
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(
                m,
                conv_mode=self.conv_init_mode,
                linear_mode=self.linear_init_mode,
                norm_mode=self.norm_init_mode,
            )

    def forward(self, x):
        """x --> Upsample --> [K x Conv2D]

        Args:
            x: (B, T, H, W, C)

        Returns:
            out: (B, T, H_new, W_new, C)
        """

        x = self.upsample(x)
        if self.num_conv_layers > 0:
            B, T, H, W, C = x.shape
            x = x.reshape([B * T, H, W, C]).transpose(perm=[0, 3, 1, 2])
            x = (
                self.conv_block(x)
                .transpose(perm=[0, 2, 3, 1])
                .reshape([B, T, H, W, -1])
            )
        return x


class InitialStackPatchMergingEncoder(paddle.nn.Layer):
    def __init__(
        self,
        num_merge: int,
        in_dim: int,
        out_dim_list: Tuple[int, ...],
        downsample_scale_list: Tuple[float, ...],
        num_conv_per_merge_list: Tuple[int, ...] = None,
        activation: str = "leaky",
        padding_type: str = "nearest",
        conv_init_mode: str = "0",
        linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(InitialStackPatchMergingEncoder, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.num_merge = num_merge
        self.in_dim = in_dim
        self.out_dim_list = out_dim_list[:num_merge]
        self.downsample_scale_list = downsample_scale_list[:num_merge]
        self.num_conv_per_merge_list = num_conv_per_merge_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]
        self.conv_block_list = paddle.nn.LayerList()
        self.patch_merge_list = paddle.nn.LayerList()
        for i in range(num_merge):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]
            downsample_scale = self.downsample_scale_list[i]
            conv_block = []
            for j in range(self.num_conv_per_merge_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(
                    paddle.nn.Conv2D(
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        in_channels=conv_in_dim,
                        out_channels=out_dim,
                    )
                )
                conv_block.append(
                    paddle.nn.GroupNorm(
                        num_groups=self.num_group_list[i], num_channels=out_dim
                    )
                )
                conv_block.append(
                    act_mod.get_activation(activation)
                    if activation != "leaky_relu"
                    else nn.LeakyReLU(NEGATIVE_SLOPE)
                )
            conv_block = paddle.nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
            patch_merge = cuboid_encoder.PatchMerging3D(
                dim=out_dim,
                out_dim=out_dim,
                padding_type=padding_type,
                downsample=(1, downsample_scale, downsample_scale),
                linear_init_mode=linear_init_mode,
                norm_init_mode=norm_init_mode,
            )
            self.patch_merge_list.append(patch_merge)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(
                m,
                conv_mode=self.conv_init_mode,
                linear_mode=self.linear_init_mode,
                norm_mode=self.norm_init_mode,
            )

    def get_out_shape_list(self, input_shape):
        out_shape_list = []
        for patch_merge in self.patch_merge_list:
            input_shape = patch_merge.get_out_shape(input_shape)
            out_shape_list.append(input_shape)
        return out_shape_list

    def forward(self, x):
        """x --> [K x Conv2D] --> PatchMerge --> ... --> [K x Conv2D] --> PatchMerge

        Args:
            x: (B, T, H, W, C)

        Returns:
            out: (B, T, H_new, W_new, C_out)
        """

        for i, (conv_block, patch_merge) in enumerate(
            zip(self.conv_block_list, self.patch_merge_list)
        ):
            B, T, H, W, C = x.shape
            if self.num_conv_per_merge_list[i] > 0:
                x = x.reshape([B * T, H, W, C]).transpose(perm=[0, 3, 1, 2])
                x = conv_block(x).transpose(perm=[0, 2, 3, 1]).reshape([B, T, H, W, -1])
            x = patch_merge(x)
        return x


class FinalStackUpsamplingDecoder(paddle.nn.Layer):
    def __init__(
        self,
        target_shape_list: Tuple[Tuple[int, ...]],
        in_dim: int,
        num_conv_per_up_list: Tuple[int, ...] = None,
        activation: str = "leaky",
        conv_init_mode: str = "0",
        linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(FinalStackUpsamplingDecoder, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.target_shape_list = target_shape_list
        self.out_dim_list = [
            target_shape[-1] for target_shape in self.target_shape_list
        ]
        self.num_upsample = len(target_shape_list)
        self.in_dim = in_dim
        self.num_conv_per_up_list = num_conv_per_up_list
        self.num_group_list = [max(1, out_dim // 4) for out_dim in self.out_dim_list]
        self.conv_block_list = paddle.nn.LayerList()
        self.upsample_list = paddle.nn.LayerList()
        for i in range(self.num_upsample):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = self.out_dim_list[i - 1]
            out_dim = self.out_dim_list[i]
            upsample = cuboid_decoder.Upsample3DLayer(
                dim=in_dim,
                out_dim=in_dim,
                target_size=target_shape_list[i][:-1],
                kernel_size=3,
                conv_init_mode=conv_init_mode,
            )
            self.upsample_list.append(upsample)
            conv_block = []
            for j in range(num_conv_per_up_list[i]):
                if j == 0:
                    conv_in_dim = in_dim
                else:
                    conv_in_dim = out_dim
                conv_block.append(
                    paddle.nn.Conv2D(
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        in_channels=conv_in_dim,
                        out_channels=out_dim,
                    )
                )
                conv_block.append(
                    paddle.nn.GroupNorm(
                        num_groups=self.num_group_list[i], num_channels=out_dim
                    )
                )
                conv_block.append(
                    act_mod.get_activation(activation)
                    if activation != "leaky_relu"
                    else nn.LeakyReLU(NEGATIVE_SLOPE)
                )
            conv_block = paddle.nn.Sequential(*conv_block)
            self.conv_block_list.append(conv_block)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(
                m,
                conv_mode=self.conv_init_mode,
                linear_mode=self.linear_init_mode,
                norm_mode=self.norm_init_mode,
            )

    @staticmethod
    def get_init_params(enc_input_shape, enc_out_shape_list, large_channel=False):
        dec_target_shape_list = list(enc_out_shape_list[:-1])[::-1] + [
            tuple(enc_input_shape)
        ]
        if large_channel:
            dec_target_shape_list_large_channel = []
            for i, enc_out_shape in enumerate(enc_out_shape_list[::-1]):
                dec_target_shape_large_channel = list(dec_target_shape_list[i])
                dec_target_shape_large_channel[-1] = enc_out_shape[-1]
                dec_target_shape_list_large_channel.append(
                    tuple(dec_target_shape_large_channel)
                )
            dec_target_shape_list = dec_target_shape_list_large_channel
        dec_in_dim = enc_out_shape_list[-1][-1]
        return dec_target_shape_list, dec_in_dim

    def forward(self, x):
        """x --> Upsample --> [K x Conv2D] --> ... --> Upsample --> [K x Conv2D]

        Args:
            x: Shape (B, T, H, W, C)

        Returns:
            out: Shape (B, T, H_new, W_new, C)
        """
        for i, (conv_block, upsample) in enumerate(
            zip(self.conv_block_list, self.upsample_list)
        ):
            x = upsample(x)
            if self.num_conv_per_up_list[i] > 0:
                B, T, H, W, C = x.shape
                x = x.reshape([B * T, H, W, C]).transpose(perm=[0, 3, 1, 2])
                x = conv_block(x).transpose(perm=[0, 2, 3, 1]).reshape([B, T, H, W, -1])
        return x


class CuboidTransformer(base.Arch):
    """Cuboid Transformer for spatiotemporal forecasting

    We adopt the Non-autoregressive encoder-decoder architecture.
    The decoder takes the multi-scale memory output from the encoder.

    The initial downsampling / upsampling layers will be
    Downsampling: [K x Conv2D --> PatchMerge]
    Upsampling: [Nearest Interpolation-based Upsample --> K x Conv2D]

    x --> downsample (optional) ---> (+pos_embed) ---> enc --> mem_l         initial_z (+pos_embed) ---> FC
                                                     |            |
                                                     |------------|
                                                           |
                                                           |
             y <--- upsample (optional) <--- dec <----------

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        input_shape (Tuple[int, ...]): The shape of the input data.
        target_shape (Tuple[int, ...]): The shape of the target data.
        base_units (int, optional): The base units. Defaults to 128.
        block_units (int, optional): The block units. Defaults to None.
        scale_alpha (float, optional): We scale up the channels based on the formula:
            - round_to(base_units * max(downsample_scale) ** units_alpha, 4). Defaults to 1.0.
        num_heads (int, optional): The number of heads. Defaults to 4.
        attn_drop (float, optional): The attention dropout. Defaults to 0.0.
        proj_drop (float, optional): The projection dropout. Defaults to 0.0.
        ffn_drop (float, optional): The ffn dropout. Defaults to 0.0.
        downsample (int, optional): The rate of downsample. Defaults to 2.
        downsample_type (str, optional): The type of downsample. Defaults to "patch_merge".
        upsample_type (str, optional): The rate of upsample. Defaults to "upsample".
        upsample_kernel_size (int, optional): The kernel size of upsample. Defaults to 3.
        enc_depth (list, optional): The depth of encoder. Defaults to [4, 4, 4].
        enc_attn_patterns (str, optional): The pattern of encoder attention. Defaults to None.
        enc_cuboid_size (list, optional): The cuboid size of encoder. Defaults to [(4, 4, 4), (4, 4, 4)].
        enc_cuboid_strategy (list, optional): The cuboid strategy of encoder. Defaults to [("l", "l", "l"), ("d", "d", "d")].
        enc_shift_size (list, optional): The shift size of encoder. Defaults to [(0, 0, 0), (0, 0, 0)].
        enc_use_inter_ffn (bool, optional): Whether to use intermediate FFN for encoder. Defaults to True.
        dec_depth (list, optional): The depth of decoder. Defaults to [2, 2].
        dec_cross_start (int, optional): The cross start of decoder. Defaults to 0.
        dec_self_attn_patterns (str, optional): The partterns of decoder. Defaults to None.
        dec_self_cuboid_size (list, optional): The cuboid size of decoder. Defaults to [(4, 4, 4), (4, 4, 4)].
        dec_self_cuboid_strategy (list, optional): The strategy of decoder. Defaults to [("l", "l", "l"), ("d", "d", "d")].
        dec_self_shift_size (list, optional): The shift size of decoder. Defaults to [(1, 1, 1), (0, 0, 0)].
        dec_cross_attn_patterns (_type_, optional): The cross attention patterns of decoder. Defaults to None.
        dec_cross_cuboid_hw (list, optional): The cuboid_hw of decoder. Defaults to [(4, 4), (4, 4)].
        dec_cross_cuboid_strategy (list, optional): The cuboid strategy of decoder. Defaults to [("l", "l", "l"), ("d", "l", "l")].
        dec_cross_shift_hw (list, optional): The shift_hw of decoder. Defaults to [(0, 0), (0, 0)].
        dec_cross_n_temporal (list, optional): The cross_n_temporal of decoder. Defaults to [1, 2].
        dec_cross_last_n_frames (int, optional): The cross_last_n_frames of decoder. Defaults to None.
        dec_use_inter_ffn (bool, optional): Whether to use intermediate FFN for decoder. Defaults to True.
        dec_hierarchical_pos_embed (bool, optional): Whether to use hierarchical pos_embed for decoder. Defaults to False.
        num_global_vectors (int, optional): The num of global vectors. Defaults to 4.
        use_dec_self_global (bool, optional): Whether to use global vector for decoder. Defaults to True.
        dec_self_update_global (bool, optional): Whether to update global vector for decoder. Defaults to True.
        use_dec_cross_global (bool, optional): Whether to use cross global vector for decoder. Defaults to True.
        use_global_vector_ffn (bool, optional): Whether to use global vector FFN. Defaults to True.
        use_global_self_attn (bool, optional): Whether to use global attentions. Defaults to False.
        separate_global_qkv (bool, optional): Whether to separate global qkv. Defaults to False.
        global_dim_ratio (int, optional): The ratio of global dim. Defaults to 1.
        self_pattern (str, optional): The pattern. Defaults to "axial".
        cross_self_pattern (str, optional): The self cross pattern. Defaults to "axial".
        cross_pattern (str, optional): The cross pattern. Defaults to "cross_1x1".
        z_init_method (str, optional): How the initial input to the decoder is initialized. Defaults to "nearest_interp".
        initial_downsample_type (str, optional): The downsample type of initial. Defaults to "conv".
        initial_downsample_activation (str, optional): The downsample activation of initial. Defaults to "leaky".
        initial_downsample_scale (int, optional): The downsample scale of initial. Defaults to 1.
        initial_downsample_conv_layers (int, optional): The conv layer of downsample of initial. Defaults to 2.
        final_upsample_conv_layers (int, optional): The conv layer of final upsample. Defaults to 2.
        initial_downsample_stack_conv_num_layers (int, optional): The num of stack conv layer of initial downsample. Defaults to 1.
        initial_downsample_stack_conv_dim_list (list, optional): The dim list of stack conv of initial downsample. Defaults to None.
        initial_downsample_stack_conv_downscale_list (list, optional): The downscale list of stack conv of initial downsample. Defaults to [1].
        initial_downsample_stack_conv_num_conv_list (list, optional): The num of stack conv list of initial downsample. Defaults to [2].
        ffn_activation (str, optional): The activation of FFN. Defaults to "leaky".
        gated_ffn (bool, optional): Whether to use gate FFN. Defaults to False.
        norm_layer (str, optional): The type of normilize. Defaults to "layer_norm".
        padding_type (str, optional): The type of padding. Defaults to "ignore".
        pos_embed_type (str, optional): The type of pos embeding. Defaults to "t+hw".
        checkpoint_level (bool, optional): Whether to use checkpoint. Defaults to True.
        use_relative_pos (bool, optional): Whether to use relative pose. Defaults to True.
        self_attn_use_final_proj (bool, optional): Whether to use final projection. Defaults to True.
        dec_use_first_self_attn (bool, optional): Whether to use first self attention for decoder. Defaults to False.
        attn_linear_init_mode (str, optional): The mode of attention linear init. Defaults to "0".
        ffn_linear_init_mode (str, optional): The mode of FFN linear init. Defaults to "0".
        conv_init_mode (str, optional): The mode of conv init. Defaults to "0".
        down_up_linear_init_mode (str, optional): The mode of downsample and upsample linear init. Defaults to "0".
        norm_init_mode (str, optional): The mode of normalization init. Defaults to "0".
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        input_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
        base_units: int = 128,
        block_units: int = None,
        scale_alpha: float = 1.0,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        downsample: int = 2,
        downsample_type: str = "patch_merge",
        upsample_type: str = "upsample",
        upsample_kernel_size: int = 3,
        enc_depth: Tuple[int, ...] = [4, 4, 4],
        enc_attn_patterns: str = None,
        enc_cuboid_size: Tuple[Tuple[int, ...], ...] = [(4, 4, 4), (4, 4, 4)],
        enc_cuboid_strategy: Tuple[Tuple[str, ...], ...] = [
            ("l", "l", "l"),
            ("d", "d", "d"),
        ],
        enc_shift_size: Tuple[Tuple[int, ...], ...] = [(0, 0, 0), (0, 0, 0)],
        enc_use_inter_ffn: str = True,
        dec_depth: Tuple[int, ...] = [2, 2],
        dec_cross_start: int = 0,
        dec_self_attn_patterns: str = None,
        dec_self_cuboid_size: Tuple[Tuple[int, ...], ...] = [(4, 4, 4), (4, 4, 4)],
        dec_self_cuboid_strategy: Tuple[Tuple[str, ...], ...] = [
            ("l", "l", "l"),
            ("d", "d", "d"),
        ],
        dec_self_shift_size: Tuple[Tuple[int, ...], ...] = [(1, 1, 1), (0, 0, 0)],
        dec_cross_attn_patterns: str = None,
        dec_cross_cuboid_hw: Tuple[Tuple[int, ...], ...] = [(4, 4), (4, 4)],
        dec_cross_cuboid_strategy: Tuple[Tuple[str, ...], ...] = [
            ("l", "l", "l"),
            ("d", "l", "l"),
        ],
        dec_cross_shift_hw: Tuple[Tuple[int, ...], ...] = [(0, 0), (0, 0)],
        dec_cross_n_temporal: Tuple[int, ...] = [1, 2],
        dec_cross_last_n_frames: int = None,
        dec_use_inter_ffn: bool = True,
        dec_hierarchical_pos_embed: bool = False,
        num_global_vectors: int = 4,
        use_dec_self_global: bool = True,
        dec_self_update_global: bool = True,
        use_dec_cross_global: bool = True,
        use_global_vector_ffn: bool = True,
        use_global_self_attn: bool = False,
        separate_global_qkv: bool = False,
        global_dim_ratio: int = 1,
        self_pattern: str = "axial",
        cross_self_pattern: str = "axial",
        cross_pattern: str = "cross_1x1",
        z_init_method: str = "nearest_interp",
        initial_downsample_type: str = "conv",
        initial_downsample_activation: str = "leaky",
        initial_downsample_scale: int = 1,
        initial_downsample_conv_layers: int = 2,
        final_upsample_conv_layers: int = 2,
        initial_downsample_stack_conv_num_layers: int = 1,
        initial_downsample_stack_conv_dim_list: Tuple[int, ...] = None,
        initial_downsample_stack_conv_downscale_list: Tuple[int, ...] = [1],
        initial_downsample_stack_conv_num_conv_list: Tuple[int, ...] = [2],
        ffn_activation: str = "leaky",
        gated_ffn: bool = False,
        norm_layer: str = "layer_norm",
        padding_type: str = "ignore",
        pos_embed_type: str = "t+hw",
        checkpoint_level: bool = True,
        use_relative_pos: bool = True,
        self_attn_use_final_proj: bool = True,
        dec_use_first_self_attn: bool = False,
        attn_linear_init_mode: str = "0",
        ffn_linear_init_mode: str = "0",
        conv_init_mode: str = "0",
        down_up_linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_up_linear_init_mode = down_up_linear_init_mode
        self.norm_init_mode = norm_init_mode
        assert len(enc_depth) == len(dec_depth)
        self.base_units = base_units
        self.num_global_vectors = num_global_vectors

        num_blocks = len(enc_depth)
        if isinstance(self_pattern, str):
            enc_attn_patterns = [self_pattern] * num_blocks

        if isinstance(cross_self_pattern, str):
            dec_self_attn_patterns = [cross_self_pattern] * num_blocks

        if isinstance(cross_pattern, str):
            dec_cross_attn_patterns = [cross_pattern] * num_blocks

        if global_dim_ratio != 1:
            assert (
                separate_global_qkv is True
            ), "Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.z_init_method = z_init_method
        assert self.z_init_method in ["zeros", "nearest_interp", "last", "mean"]
        self.input_shape = input_shape
        self.target_shape = target_shape
        T_in, H_in, W_in, C_in = input_shape
        T_out, H_out, W_out, C_out = target_shape
        assert H_in == H_out and W_in == W_out
        if self.num_global_vectors > 0:
            init_data = paddle.zeros(
                (self.num_global_vectors, global_dim_ratio * base_units)
            )
            self.init_global_vectors = paddle.create_parameter(
                shape=init_data.shape,
                dtype=init_data.dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )

            self.init_global_vectors.stop_gradient = not True
        new_input_shape = self.get_initial_encoder_final_decoder(
            initial_downsample_scale=initial_downsample_scale,
            initial_downsample_type=initial_downsample_type,
            activation=initial_downsample_activation,
            initial_downsample_conv_layers=initial_downsample_conv_layers,
            final_upsample_conv_layers=final_upsample_conv_layers,
            padding_type=padding_type,
            initial_downsample_stack_conv_num_layers=initial_downsample_stack_conv_num_layers,
            initial_downsample_stack_conv_dim_list=initial_downsample_stack_conv_dim_list,
            initial_downsample_stack_conv_downscale_list=initial_downsample_stack_conv_downscale_list,
            initial_downsample_stack_conv_num_conv_list=initial_downsample_stack_conv_num_conv_list,
        )
        T_in, H_in, W_in, _ = new_input_shape
        self.encoder = cuboid_encoder.CuboidTransformerEncoder(
            input_shape=(T_in, H_in, W_in, base_units),
            base_units=base_units,
            block_units=block_units,
            scale_alpha=scale_alpha,
            depth=enc_depth,
            downsample=downsample,
            downsample_type=downsample_type,
            block_attn_patterns=enc_attn_patterns,
            block_cuboid_size=enc_cuboid_size,
            block_strategy=enc_cuboid_strategy,
            block_shift_size=enc_shift_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            gated_ffn=gated_ffn,
            ffn_activation=ffn_activation,
            norm_layer=norm_layer,
            use_inter_ffn=enc_use_inter_ffn,
            padding_type=padding_type,
            use_global_vector=num_global_vectors > 0,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            down_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.enc_pos_embed = cuboid_decoder.PosEmbed(
            embed_dim=base_units, typ=pos_embed_type, maxH=H_in, maxW=W_in, maxT=T_in
        )
        mem_shapes = self.encoder.get_mem_shapes()
        self.z_proj = paddle.nn.Linear(
            in_features=mem_shapes[-1][-1], out_features=mem_shapes[-1][-1]
        )
        self.dec_pos_embed = cuboid_decoder.PosEmbed(
            embed_dim=mem_shapes[-1][-1],
            typ=pos_embed_type,
            maxT=T_out,
            maxH=mem_shapes[-1][1],
            maxW=mem_shapes[-1][2],
        )
        self.decoder = cuboid_decoder.CuboidTransformerDecoder(
            target_temporal_length=T_out,
            mem_shapes=mem_shapes,
            cross_start=dec_cross_start,
            depth=dec_depth,
            upsample_type=upsample_type,
            block_self_attn_patterns=dec_self_attn_patterns,
            block_self_cuboid_size=dec_self_cuboid_size,
            block_self_shift_size=dec_self_shift_size,
            block_self_cuboid_strategy=dec_self_cuboid_strategy,
            block_cross_attn_patterns=dec_cross_attn_patterns,
            block_cross_cuboid_hw=dec_cross_cuboid_hw,
            block_cross_shift_hw=dec_cross_shift_hw,
            block_cross_cuboid_strategy=dec_cross_cuboid_strategy,
            block_cross_n_temporal=dec_cross_n_temporal,
            cross_last_n_frames=dec_cross_last_n_frames,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            upsample_kernel_size=upsample_kernel_size,
            ffn_activation=ffn_activation,
            gated_ffn=gated_ffn,
            norm_layer=norm_layer,
            use_inter_ffn=dec_use_inter_ffn,
            max_temporal_relative=T_in + T_out,
            padding_type=padding_type,
            hierarchical_pos_embed=dec_hierarchical_pos_embed,
            pos_embed_type=pos_embed_type,
            use_self_global=num_global_vectors > 0 and use_dec_self_global,
            self_update_global=dec_self_update_global,
            use_cross_global=num_global_vectors > 0 and use_dec_cross_global,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            checkpoint_level=checkpoint_level,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            use_first_self_attn=dec_use_first_self_attn,
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            conv_init_mode=conv_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )
        self.reset_parameters()

    def get_initial_encoder_final_decoder(
        self,
        initial_downsample_type,
        activation,
        initial_downsample_scale,
        initial_downsample_conv_layers,
        final_upsample_conv_layers,
        padding_type,
        initial_downsample_stack_conv_num_layers,
        initial_downsample_stack_conv_dim_list,
        initial_downsample_stack_conv_downscale_list,
        initial_downsample_stack_conv_num_conv_list,
    ):
        T_in, H_in, W_in, C_in = self.input_shape
        T_out, H_out, W_out, C_out = self.target_shape
        self.initial_downsample_type = initial_downsample_type
        if self.initial_downsample_type == "conv":
            if isinstance(initial_downsample_scale, int):
                initial_downsample_scale = (
                    1,
                    initial_downsample_scale,
                    initial_downsample_scale,
                )
            elif len(initial_downsample_scale) == 2:
                initial_downsample_scale = 1, *initial_downsample_scale
            elif len(initial_downsample_scale) == 3:
                initial_downsample_scale = tuple(initial_downsample_scale)
            else:
                raise NotImplementedError(
                    f"initial_downsample_scale {initial_downsample_scale} format not supported!"
                )
            self.initial_encoder = InitialEncoder(
                dim=C_in,
                out_dim=self.base_units,
                downsample_scale=initial_downsample_scale,
                num_conv_layers=initial_downsample_conv_layers,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )

            self.final_decoder = FinalDecoder(
                dim=self.base_units,
                target_thw=(T_out, H_out, W_out),
                num_conv_layers=final_upsample_conv_layers,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            new_input_shape = self.initial_encoder.patch_merge.get_out_shape(
                self.input_shape
            )
            self.dec_final_proj = paddle.nn.Linear(
                in_features=self.base_units, out_features=C_out
            )
        elif self.initial_downsample_type == "stack_conv":
            if initial_downsample_stack_conv_dim_list is None:
                initial_downsample_stack_conv_dim_list = [
                    self.base_units
                ] * initial_downsample_stack_conv_num_layers
            self.initial_encoder = InitialStackPatchMergingEncoder(
                num_merge=initial_downsample_stack_conv_num_layers,
                in_dim=C_in,
                out_dim_list=initial_downsample_stack_conv_dim_list,
                downsample_scale_list=initial_downsample_stack_conv_downscale_list,
                num_conv_per_merge_list=initial_downsample_stack_conv_num_conv_list,
                padding_type=padding_type,
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            initial_encoder_out_shape_list = self.initial_encoder.get_out_shape_list(
                self.target_shape
            )
            (
                dec_target_shape_list,
                dec_in_dim,
            ) = FinalStackUpsamplingDecoder.get_init_params(
                enc_input_shape=self.target_shape,
                enc_out_shape_list=initial_encoder_out_shape_list,
                large_channel=True,
            )
            self.final_decoder = FinalStackUpsamplingDecoder(
                target_shape_list=dec_target_shape_list,
                in_dim=dec_in_dim,
                num_conv_per_up_list=initial_downsample_stack_conv_num_conv_list[::-1],
                activation=activation,
                conv_init_mode=self.conv_init_mode,
                linear_init_mode=self.down_up_linear_init_mode,
                norm_init_mode=self.norm_init_mode,
            )
            self.dec_final_proj = paddle.nn.Linear(
                in_features=dec_target_shape_list[-1][-1], out_features=C_out
            )
            new_input_shape = self.initial_encoder.get_out_shape_list(self.input_shape)[
                -1
            ]
        else:
            raise NotImplementedError(f"{self.initial_downsample_type} is invalid.")
        self.input_shape_after_initial_downsample = new_input_shape
        T_in, H_in, W_in, _ = new_input_shape
        return new_input_shape

    def reset_parameters(self):
        if self.num_global_vectors > 0:
            self.init_global_vectors = initializer.trunc_normal_(
                self.init_global_vectors, std=0.02
            )
        if hasattr(self.initial_encoder, "reset_parameters"):
            self.initial_encoder.reset_parameters()
        else:
            cuboid_utils.apply_initialization(
                self.initial_encoder,
                conv_mode=self.conv_init_mode,
                linear_mode=self.down_up_linear_init_mode,
                norm_mode=self.norm_init_mode,
            )
        if hasattr(self.final_decoder, "reset_parameters"):
            self.final_decoder.reset_parameters()
        else:
            cuboid_utils.apply_initialization(
                self.final_decoder,
                conv_mode=self.conv_init_mode,
                linear_mode=self.down_up_linear_init_mode,
                norm_mode=self.norm_init_mode,
            )
        cuboid_utils.apply_initialization(
            self.dec_final_proj, linear_mode=self.down_up_linear_init_mode
        )
        self.encoder.reset_parameters()
        self.enc_pos_embed.reset_parameters()
        self.decoder.reset_parameters()
        self.dec_pos_embed.reset_parameters()
        cuboid_utils.apply_initialization(self.z_proj, linear_mode="0")

    def get_initial_z(self, final_mem, T_out):
        B = final_mem.shape[0]
        if self.z_init_method == "zeros":
            z_shape = list((1, T_out)) + final_mem.shape[2:]
            initial_z = paddle.zeros(shape=z_shape, dtype=final_mem.dtype)
            initial_z = self.z_proj(self.dec_pos_embed(initial_z)).expand(
                shape=[B, -1, -1, -1, -1]
            )
        elif self.z_init_method == "nearest_interp":
            initial_z = paddle.nn.functional.interpolate(
                x=final_mem.transpose(perm=[0, 4, 1, 2, 3]),
                size=(T_out, final_mem.shape[2], final_mem.shape[3]),
            ).transpose(perm=[0, 2, 3, 4, 1])
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == "last":
            initial_z = paddle.broadcast_to(
                x=final_mem[:, -1:, :, :, :], shape=(B, T_out) + final_mem.shape[2:]
            )
            initial_z = self.z_proj(initial_z)
        elif self.z_init_method == "mean":
            initial_z = paddle.broadcast_to(
                x=final_mem.mean(axis=1, keepdims=True),
                shape=(B, T_out) + final_mem.shape[2:],
            )
            initial_z = self.z_proj(initial_z)
        else:
            raise NotImplementedError
        return initial_z

    def forward(self, x: "paddle.Tensor", verbose: bool = False) -> "paddle.Tensor":
        """
        Args:
            x (paddle.Tensor): Tensor with shape (B, T, H, W, C).
            verbose (bool): if True, print intermediate shapes.

        Returns:
            out (paddle.Tensor): The output Shape (B, T_out, H, W, C_out)
        """

        x = self.concat_to_tensor(x, self.input_keys)
        flag_ndim = x.ndim
        if flag_ndim == 6:
            x = x.reshape([-1, *x.shape[2:]])
        B, _, _, _, _ = x.shape

        T_out = self.target_shape[0]
        x = self.initial_encoder(x)
        x = self.enc_pos_embed(x)

        if self.num_global_vectors > 0:
            init_global_vectors = self.init_global_vectors.expand(
                shape=[
                    B,
                    self.num_global_vectors,
                    self.global_dim_ratio * self.base_units,
                ]
            )
            mem_l, mem_global_vector_l = self.encoder(x, init_global_vectors)
        else:
            mem_l = self.encoder(x)

        if verbose:
            for i, mem in enumerate(mem_l):
                print(f"mem[{i}].shape = {mem.shape}")
        initial_z = self.get_initial_z(final_mem=mem_l[-1], T_out=T_out)

        if self.num_global_vectors > 0:
            dec_out = self.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            dec_out = self.decoder(initial_z, mem_l)

        dec_out = self.final_decoder(dec_out)

        out = self.dec_final_proj(dec_out)
        if flag_ndim == 6:
            out = out.reshape([-1, *out.shape])
        return {key: out for key in self.output_keys}

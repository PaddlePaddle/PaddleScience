import functools
from typing import Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn

from ppsci.utils import initializer


def round_to(dat, c):
    return dat + (dat - dat % c) % c


class RMSNorm(paddle.nn.Layer):
    """Root Mean Square Layer Normalization proposed in "[NeurIPS2019] Root Mean Square Layer Normalization"

    Args:
        d (Optional[int]): The model size.
        p (float, optional): The partial RMSNorm, valid value [0, 1]. Defaults to -1.0.
        eps (float, optional): The epsilon value. Defaults to 1e-08.
        bias (bool, optional): Whether use bias term for RMSNorm,
            because RMSNorm doesn't enforce re-centering invariance.Defaults to False.
    """

    def __init__(
        self,
        d: Tuple[int, ...],
        p: float = -1.0,
        eps: float = 1e-08,
        bias: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        init_data = paddle.ones(d)
        self.scale = paddle.create_parameter(
            shape=init_data.shape,
            dtype=init_data.dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.scale.stop_gradient = False
        self.add_parameter(name="scale", parameter=self.scale)
        if self.bias:
            init_data = paddle.zeros(d)
            self.offset = paddle.create_parameter(
                shape=init_data.shape,
                dtype=init_data.dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.offset.stop_gradient = False
            self.add_parameter(name="offset", parameter=self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(p=2, axis=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = paddle.split(
                x=x, num_or_sections=[partial_size, self.d - partial_size], axis=-1
            )
            norm_x = partial_x.norm(p=2, axis=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


def get_norm_layer(
    normalization: str = "layer_norm",
    axis: int = -1,
    epsilon: float = 1e-05,
    in_channels: int = 0,
    **kwargs,
):
    """Get the normalization layer based on the provided type

    Args:
        normalization (str): The type of the layer normalization from ['layer_norm'].
        axis (float): The axis to normalize the.
        epsilon (float): The epsilon of the normalization layer.
        in_channels (int): Input channel.

    Returns:
        norm_layer (norm): The layer normalization layer.
    """

    if isinstance(normalization, str):
        if normalization == "layer_norm":
            assert in_channels > 0
            assert axis == -1
            norm_layer = paddle.nn.LayerNorm(
                normalized_shape=in_channels, epsilon=epsilon, **kwargs
            )
        elif normalization == "rms_norm":
            assert axis == -1
            norm_layer = RMSNorm(d=in_channels, epsilon=epsilon, **kwargs)
        else:
            raise NotImplementedError(
                "normalization={} is not supported".format(normalization)
            )
        return norm_layer
    elif normalization is None:
        return paddle.nn.Identity()
    else:
        raise NotImplementedError("The type of normalization must be str")


def generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    assert padding_type in ["zeros", "ignore", "nearest"]
    B, T, H, W, C = x.shape
    if padding_type == "nearest":
        return paddle.nn.functional.interpolate(
            x=x.transpose(perm=[0, 4, 1, 2, 3]), size=(T + pad_t, H + pad_h, W + pad_w)
        ).transpose(perm=[0, 2, 3, 4, 1])
    elif t_pad_left:
        return F.pad(x, [0, 0, 0, pad_w, 0, pad_h, pad_t, 0], data_format="NDHWC")
    else:
        data_pad = F.pad(
            x, [0, 0, pad_t, 0, pad_h, 0, pad_w, 0, 0, 0], data_format="NDHWC"
        )
        data_pad = paddle.concat(
            [data_pad[:, pad_t:, ...], data_pad[:, :pad_t, ...]], axis=1
        )
        return data_pad


def generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    assert padding_type in ["zeros", "ignore", "nearest"]
    B, T, H, W, C = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    if padding_type == "nearest":
        return paddle.nn.functional.interpolate(
            x=x.transpose(perm=[0, 4, 1, 2, 3]), size=(T - pad_t, H - pad_h, W - pad_w)
        ).transpose(perm=[0, 2, 3, 4, 1])
    else:
        return x[:, : T - pad_t, : H - pad_h, : W - pad_w, :]


def apply_initialization(
    m: paddle.nn.Layer,
    linear_mode: str = "0",
    conv_mode: str = "0",
    norm_mode: str = "0",
    embed_mode: str = "0",
):
    if isinstance(m, paddle.nn.Linear):
        if linear_mode in ("0",):
            m.weight = initializer.kaiming_normal_(m.weight, nonlinearity="linear")
        elif linear_mode in ("1",):
            m.weight = initializer.kaiming_normal_(
                m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
            )
        else:
            raise NotImplementedError(f"{linear_mode} is invalid.")
        if hasattr(m, "bias") and m.bias is not None:
            m.bias = initializer.zeros_(m.bias)
    elif isinstance(
        m,
        (
            paddle.nn.Conv2D,
            paddle.nn.Conv3D,
            paddle.nn.Conv2DTranspose,
            paddle.nn.Conv3DTranspose,
        ),
    ):
        if conv_mode in ("0",):
            m.weight = initializer.kaiming_normal_(
                m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
            )
        else:
            raise NotImplementedError(f"{conv_mode} is invalid.")
        if hasattr(m, "bias") and m.bias is not None:
            m.bias = initializer.zeros_(m.bias)
    elif isinstance(m, paddle.nn.LayerNorm):
        if norm_mode in ("0",):
            m.weight = initializer.zeros_(m.weight)
            m.bias = initializer.zeros_(m.bias)
        else:
            raise NotImplementedError(f"{norm_mode} is invalid.")
    elif isinstance(m, paddle.nn.GroupNorm):
        if norm_mode in ("0",):
            m.weight = initializer.ones_(m.weight)
            m.bias = initializer.zeros_(m.bias)
        else:
            raise NotImplementedError(f"{norm_mode} is invalid.")
    elif isinstance(m, paddle.nn.Embedding):
        if embed_mode in ("0",):
            m.weight.data = initializer.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError(f"{embed_mode} is invalid.")

    else:
        pass


class CuboidSelfAttentionPatterns:
    def __init__(self):
        super().__init__()
        self.patterns = {}
        self.patterns = {
            "full": self.full_attention,
            "axial": self.axial,
            "divided_st": self.divided_space_time,
        }
        for p in [1, 2, 4, 8, 10]:
            for m in [1, 2, 4, 8, 16, 32]:
                key = f"video_swin_{p}x{m}"
                self.patterns[key] = functools.partial(self.video_swin, P=p, M=m)

        for m in [1, 2, 4, 8, 16, 32]:
            key = f"spatial_lg_{m}"
            self.patterns[key] = functools.partial(self.spatial_lg_v1, M=m)

        for k in [2, 4, 8]:
            key = f"axial_space_dilate_{k}"
            self.patterns[key] = functools.partial(self.axial_space_dilate_K, K=k)

    def get(self, pattern_name):
        return self.patterns[pattern_name]

    def full_attention(self, input_shape):
        T, H, W, _ = input_shape
        cuboid_size = [(T, H, W)]
        strategy = [("l", "l", "l")]
        shift_size = [(0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def axial(self, input_shape):
        """Axial attention proposed in https://arxiv.org/abs/1912.12180

        Args:
            input_shape (Tuple[int,...]): The shape of the input tensor, T H W.

        Returns:
            cuboid_size (Tuple[int,...]): The size of cuboid.
            strategy (Tuple[str,...]): The strategy of the attention.
            shift_size (Tuple[int,...]): The shift size of the attention.
        """

        T, H, W, _ = input_shape
        cuboid_size = [(T, 1, 1), (1, H, 1), (1, 1, W)]
        strategy = [("l", "l", "l"), ("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def divided_space_time(self, input_shape):
        T, H, W, _ = input_shape
        cuboid_size = [(T, 1, 1), (1, H, W)]
        strategy = [("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def video_swin(self, input_shape, P=2, M=4):
        """Adopt the strategy in Video SwinTransformer https://arxiv.org/pdf/2106.13230.pdf"""
        T, H, W, _ = input_shape
        P = min(P, T)
        M = min(M, H, W)
        cuboid_size = [(P, M, M), (P, M, M)]
        strategy = [("l", "l", "l"), ("l", "l", "l")]
        shift_size = [(0, 0, 0), (P // 2, M // 2, M // 2)]
        return cuboid_size, strategy, shift_size

    def spatial_lg_v1(self, input_shape, M=4):
        T, H, W, _ = input_shape
        if H <= M and W <= M:
            cuboid_size = [(T, 1, 1), (1, H, W)]
            strategy = [("l", "l", "l"), ("l", "l", "l")]
            shift_size = [(0, 0, 0), (0, 0, 0)]
        else:
            cuboid_size = [(T, 1, 1), (1, M, M), (1, M, M)]
            strategy = [("l", "l", "l"), ("l", "l", "l"), ("d", "d", "d")]
            shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size

    def axial_space_dilate_K(self, input_shape, K=2):
        T, H, W, _ = input_shape
        K = min(K, H, W)
        cuboid_size = [
            (T, 1, 1),
            (1, H // K, 1),
            (1, H // K, 1),
            (1, 1, W // K),
            (1, 1, W // K),
        ]
        strategy = [
            ("l", "l", "l"),
            ("d", "d", "d"),
            ("l", "l", "l"),
            ("d", "d", "d"),
            ("l", "l", "l"),
        ]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        return cuboid_size, strategy, shift_size


class CuboidCrossAttentionPatterns:
    def __init__(self):
        super().__init__()
        self.patterns = {}
        for k in [1, 2, 4, 8]:
            key1 = f"cross_{k}x{k}"
            key2 = f"cross_{k}x{k}_lg"
            key3 = f"cross_{k}x{k}_heter"
            self.patterns[key1] = functools.partial(self.cross_KxK, K=k)
            self.patterns[key2] = functools.partial(self.cross_KxK_lg, K=k)
            self.patterns[key3] = functools.partial(self.cross_KxK_heter, K=k)

    def get(self, pattern_name):
        return self.patterns[pattern_name]

    def cross_KxK(self, mem_shape, K):
        T_mem, H, W, _ = mem_shape
        K = min(K, H, W)
        cuboid_hw = [(K, K)]
        shift_hw = [(0, 0)]
        strategy = [("l", "l", "l")]
        n_temporal = [1]
        return cuboid_hw, shift_hw, strategy, n_temporal

    def cross_KxK_lg(self, mem_shape, K):
        T_mem, H, W, _ = mem_shape
        K = min(K, H, W)
        cuboid_hw = [(K, K), (K, K)]
        shift_hw = [(0, 0), (0, 0)]
        strategy = [("l", "l", "l"), ("d", "d", "d")]
        n_temporal = [1, 1]
        return cuboid_hw, shift_hw, strategy, n_temporal

    def cross_KxK_heter(self, mem_shape, K):
        T_mem, H, W, _ = mem_shape
        K = min(K, H, W)
        cuboid_hw = [(K, K), (K, K), (K, K)]
        shift_hw = [(0, 0), (0, 0), (K // 2, K // 2)]
        strategy = [("l", "l", "l"), ("d", "d", "d"), ("l", "l", "l")]
        n_temporal = [1, 1, 1]
        return cuboid_hw, shift_hw, strategy, n_temporal


CuboidSelfAttentionPatterns = CuboidSelfAttentionPatterns()
CuboidCrossAttentionPatterns = CuboidCrossAttentionPatterns()

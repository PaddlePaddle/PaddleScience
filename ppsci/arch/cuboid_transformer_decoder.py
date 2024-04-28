from functools import lru_cache
from typing import Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet

import ppsci.arch.cuboid_transformer_encoder as cuboid_encoder
import ppsci.arch.cuboid_transformer_utils as cuboid_utils
from ppsci.utils import initializer


class PosEmbed(paddle.nn.Layer):
    """pose embeding

    Args:
        embed_dim (int): The dimension of embeding.
        maxT (int): The embeding max time.
        maxH (int): The embeding max height.
        maxW (int): The embeding max width.
        typ (str):
            The type of the positional embedding.
            - t+h+w:
                Embed the spatial position to embeddings
            - t+hw:
                Embed the spatial position to embeddings
    """

    def __init__(self, embed_dim, maxT, maxH, maxW, typ: str = "t+h+w"):
        super(PosEmbed, self).__init__()
        self.typ = typ
        assert self.typ in ["t+h+w", "t+hw"]
        self.maxT = maxT
        self.maxH = maxH
        self.maxW = maxW
        self.embed_dim = embed_dim
        if self.typ == "t+h+w":
            self.T_embed = paddle.nn.Embedding(
                num_embeddings=maxT, embedding_dim=embed_dim
            )
            self.H_embed = paddle.nn.Embedding(
                num_embeddings=maxH, embedding_dim=embed_dim
            )
            self.W_embed = paddle.nn.Embedding(
                num_embeddings=maxW, embedding_dim=embed_dim
            )
        elif self.typ == "t+hw":
            self.T_embed = paddle.nn.Embedding(
                num_embeddings=maxT, embedding_dim=embed_dim
            )
            self.HW_embed = paddle.nn.Embedding(
                num_embeddings=maxH * maxW, embedding_dim=embed_dim
            )
        else:
            raise NotImplementedError(f"{self.typ} is invalid.")
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(m, embed_mode="0")

    def forward(self, x):
        """
        Args:
            x : Shape (B, T, H, W, C)

        Returns:
            out : the x + positional embeddings
        """

        _, T, H, W, _ = x.shape
        t_idx = paddle.arange(end=T)
        h_idx = paddle.arange(end=H)
        w_idx = paddle.arange(end=W)
        if self.typ == "t+h+w":
            return (
                x
                + self.T_embed(t_idx).reshape([T, 1, 1, self.embed_dim])
                + self.H_embed(h_idx).reshape([1, H, 1, self.embed_dim])
                + self.W_embed(w_idx).reshape([1, 1, W, self.embed_dim])
            )
        elif self.typ == "t+hw":
            spatial_idx = h_idx.unsqueeze(axis=-1) * self.maxW + w_idx
            return (
                x
                + self.T_embed(t_idx).reshape([T, 1, 1, self.embed_dim])
                + self.HW_embed(spatial_idx)
            )
        else:
            raise NotImplementedError(f"{self.typ} is invalid.")


@lru_cache()
def compute_cuboid_cross_attention_mask(
    T_x, T_mem, H, W, n_temporal, cuboid_hw, shift_hw, strategy, padding_type, device
):
    pad_t_mem = (n_temporal - T_mem % n_temporal) % n_temporal
    pad_t_x = (n_temporal - T_x % n_temporal) % n_temporal
    pad_h = (cuboid_hw[0] - H % cuboid_hw[0]) % cuboid_hw[0]
    pad_w = (cuboid_hw[1] - W % cuboid_hw[1]) % cuboid_hw[1]
    mem_cuboid_size = ((T_mem + pad_t_mem) // n_temporal,) + cuboid_hw
    x_cuboid_size = ((T_x + pad_t_x) // n_temporal,) + cuboid_hw
    if pad_t_mem > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == "ignore":
            mem_mask = paddle.ones(shape=(1, T_mem, H, W, 1), dtype="bool")
            mem_mask = F.pad(
                mem_mask, [0, 0, 0, pad_w, 0, pad_h, pad_t_mem, 0], data_format="NDHWC"
            )
    else:
        mem_mask = paddle.ones(
            shape=(1, T_mem + pad_t_mem, H + pad_h, W + pad_w, 1), dtype="bool"
        )
    if pad_t_x > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == "ignore":
            x_mask = paddle.ones(shape=(1, T_x, H, W, 1), dtype="bool")
            x_mask = F.pad(
                x_mask, [0, 0, 0, pad_w, 0, pad_h, 0, pad_t_x], data_format="NDHWC"
            )
    else:
        x_mask = paddle.ones(
            shape=(1, T_x + pad_t_x, H + pad_h, W + pad_w, 1), dtype="bool"
        )
    if any(i > 0 for i in shift_hw):
        if padding_type == "ignore":
            x_mask = paddle.roll(
                x=x_mask, shifts=(-shift_hw[0], -shift_hw[1]), axis=(2, 3)
            )
            mem_mask = paddle.roll(
                x=mem_mask, shifts=(-shift_hw[0], -shift_hw[1]), axis=(2, 3)
            )
    x_mask = cuboid_encoder.cuboid_reorder(x_mask, x_cuboid_size, strategy=strategy)
    x_mask = x_mask.squeeze(axis=-1).squeeze(axis=0)
    num_cuboids, x_cuboid_volume = x_mask.shape
    mem_mask = cuboid_encoder.cuboid_reorder(
        mem_mask, mem_cuboid_size, strategy=strategy
    )
    mem_mask = mem_mask.squeeze(axis=-1).squeeze(axis=0)
    _, mem_cuboid_volume = mem_mask.shape
    shift_mask = np.zeros(shape=(1, n_temporal, H + pad_h, W + pad_w, 1))
    cnt = 0
    for h in (
        slice(-cuboid_hw[0]),
        slice(-cuboid_hw[0], -shift_hw[0]),
        slice(-shift_hw[0], None),
    ):
        for w in (
            slice(-cuboid_hw[1]),
            slice(-cuboid_hw[1], -shift_hw[1]),
            slice(-shift_hw[1], None),
        ):
            shift_mask[:, :, h, w, :] = cnt
            cnt += 1
    shift_mask = paddle.to_tensor(shift_mask)
    shift_mask = cuboid_encoder.cuboid_reorder(
        shift_mask, (1,) + cuboid_hw, strategy=strategy
    )
    shift_mask = shift_mask.squeeze(axis=-1).squeeze(axis=0)
    shift_mask = shift_mask.unsqueeze(axis=1) - shift_mask.unsqueeze(axis=2) == 0
    bh_bw = cuboid_hw[0] * cuboid_hw[1]
    attn_mask = (
        shift_mask.reshape((num_cuboids, 1, bh_bw, 1, bh_bw))
        * x_mask.reshape((num_cuboids, -1, bh_bw, 1, 1))
        * mem_mask.reshape([num_cuboids, 1, 1, -1, bh_bw])
    )
    attn_mask = attn_mask.reshape([num_cuboids, x_cuboid_volume, mem_cuboid_volume])
    return attn_mask


class CuboidCrossAttentionLayer(paddle.nn.Layer):
    """Implements the cuboid cross attention.

    The idea of Cuboid Cross Attention is to extend the idea of cuboid self attention to work for the
    encoder-decoder-type cross attention.

    Assume that there is a memory tensor with shape (T1, H, W, C) and another query tensor with shape (T2, H, W, C),

    Here, we decompose the query tensor and the memory tensor into the same number of cuboids and attend the cuboid in
    the query tensor with the corresponding cuboid in the memory tensor.

    For the height and width axes, we reuse the grid decomposition techniques described in the cuboid self-attention.
    For the temporal axis, the layer supports the "n_temporal" parameter, that controls the number of cuboids we can
    get after cutting the tensors. For example, if the temporal dilation is 2, both the query and
    memory will be decomposed into 2 cuboids along the temporal axis. Like in the Cuboid Self-attention,
    we support "local" and "dilated" decomposition strategy.

    The complexity of the layer is O((T2 / n_t * Bh * Bw) * (T1 / n_t * Bh * Bw) * n_t (H / Bh) (W / Bw)) = O(T2 * T1 / n_t H W Bh Bw)

    Args:
        dim (int): The dimention of input tensor.
        num_heads (int): The number of head.
        n_temporal (int, optional): The num of temporal. Defaults to 1.
        cuboid_hw (tuple, optional): The height and width of cuboid. Defaults to (7, 7).
        shift_hw (tuple, optional): The height and width of shift. Defaults to (0, 0).
        strategy (tuple, optional): The strategy. Defaults to ("d", "l", "l").
        padding_type (str, optional): The type of padding. Defaults to "ignore".
        cross_last_n_frames (int, optional): The cross_last_n_frames of decoder. Defaults to None.
        qkv_bias (bool, optional): Whether to enable bias in calculating qkv attention. Defaults to False.
        qk_scale (float, optional): Whether to enable scale factor when calculating the attention. Defaults to None.
        attn_drop (float, optional): The attention dropout. Defaults to 0.0.
        proj_drop (float, optional): The projrction dropout. Defaults to 0.0.
        max_temporal_relative (int, optional): The max temporal. Defaults to 50.
        norm_layer (str, optional): The normalization layer. Defaults to "layer_norm".
        use_global_vector (bool, optional): Whether to use the global vector or not. Defaults to True.
        separate_global_qkv (bool, optional): Whether to use different network to calc q_global, k_global, v_global. Defaults to False.
        global_dim_ratio (int, optional): The dim (channels) of global vectors is `global_dim_ratio*dim`. Defaults to 1.
        checkpoint_level (int, optional): Whether to enable gradient checkpointing. Defaults to 1.
        use_relative_pos (bool, optional):  Whether to use relative pos. Defaults to True.
        attn_linear_init_mode (str, optional): The mode of attention linear initialization. Defaults to "0".
        ffn_linear_init_mode (str, optional): The mode of FFN linear initialization. Defaults to "0".
        norm_init_mode (str, optional): The mode of normalization initialization. Defaults to "0".
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_temporal: int = 1,
        cuboid_hw: Tuple[int, ...] = (7, 7),
        shift_hw: Tuple[int, ...] = (0, 0),
        strategy: Tuple[str, ...] = ("d", "l", "l"),
        padding_type: str = "ignore",
        cross_last_n_frames: int = None,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        max_temporal_relative: int = 50,
        norm_layer: str = "layer_norm",
        use_global_vector: bool = True,
        separate_global_qkv: bool = False,
        global_dim_ratio: int = 1,
        checkpoint_level: int = 1,
        use_relative_pos: bool = True,
        attn_linear_init_mode: str = "0",
        ffn_linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(CuboidCrossAttentionLayer, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        self.num_heads = num_heads
        self.n_temporal = n_temporal
        assert n_temporal > 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        shift_hw = list(shift_hw)
        if strategy[1] == "d":
            shift_hw[0] = 0
        if strategy[2] == "d":
            shift_hw[1] = 0
        self.cuboid_hw = cuboid_hw
        self.shift_hw = tuple(shift_hw)
        self.strategy = strategy
        self.padding_type = padding_type
        self.max_temporal_relative = max_temporal_relative
        self.cross_last_n_frames = cross_last_n_frames
        self.use_relative_pos = use_relative_pos
        self.use_global_vector = use_global_vector
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1 and separate_global_qkv is False:
            raise ValueError(
                "Setting global_dim_ratio != 1 requires separate_global_qkv == True."
            )
        self.global_dim_ratio = global_dim_ratio
        if self.padding_type not in ["ignore", "zeros", "nearest"]:
            raise ValueError('padding_type should be ["ignore", "zeros", "nearest"]')
        if use_relative_pos:
            init_data = paddle.zeros(
                (
                    (2 * max_temporal_relative - 1)
                    * (2 * cuboid_hw[0] - 1)
                    * (2 * cuboid_hw[1] - 1),
                    num_heads,
                )
            )
            self.relative_position_bias_table = paddle.create_parameter(
                shape=init_data.shape,
                dtype=init_data.dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.relative_position_bias_table.stop_gradient = not True
            self.relative_position_bias_table = initializer.trunc_normal_(
                self.relative_position_bias_table, std=0.02
            )

            coords_t = paddle.arange(end=max_temporal_relative)
            coords_h = paddle.arange(end=self.cuboid_hw[0])
            coords_w = paddle.arange(end=self.cuboid_hw[1])
            coords = paddle.stack(x=paddle.meshgrid(coords_t, coords_h, coords_w))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.transpose(perm=[1, 2, 0])
            relative_coords[:, :, 0] += max_temporal_relative - 1
            relative_coords[:, :, 1] += self.cuboid_hw[0] - 1
            relative_coords[:, :, 2] += self.cuboid_hw[1] - 1
            relative_position_index = (
                relative_coords[:, :, 0]
                * (2 * self.cuboid_hw[0] - 1)
                * (2 * self.cuboid_hw[1] - 1)
                + relative_coords[:, :, 1] * (2 * self.cuboid_hw[1] - 1)
                + relative_coords[:, :, 2]
            )
            self.register_buffer(
                name="relative_position_index", tensor=relative_position_index
            )
        self.q_proj = paddle.nn.Linear(
            in_features=dim, out_features=dim, bias_attr=qkv_bias
        )
        self.kv_proj = paddle.nn.Linear(
            in_features=dim, out_features=dim * 2, bias_attr=qkv_bias
        )
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop)
        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = paddle.nn.Linear(
                    in_features=dim, out_features=dim, bias_attr=qkv_bias
                )
                self.l2g_global_kv_net = paddle.nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim * 2,
                    bias_attr=qkv_bias,
                )
        self.norm = cuboid_utils.get_norm_layer(norm_layer, in_channels=dim)
        self._checkpoint_level = checkpoint_level
        self.reset_parameters()

    def reset_parameters(self):
        cuboid_utils.apply_initialization(
            self.q_proj, linear_mode=self.attn_linear_init_mode
        )
        cuboid_utils.apply_initialization(
            self.kv_proj, linear_mode=self.attn_linear_init_mode
        )
        cuboid_utils.apply_initialization(
            self.proj, linear_mode=self.ffn_linear_init_mode
        )
        cuboid_utils.apply_initialization(self.norm, norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                cuboid_utils.apply_initialization(
                    self.l2g_q_net, linear_mode=self.attn_linear_init_mode
                )
                cuboid_utils.apply_initialization(
                    self.l2g_global_kv_net, linear_mode=self.attn_linear_init_mode
                )

    def forward(self, x, mem, mem_global_vectors=None):
        """Calculate the forward

        Along the temporal axis, we pad the mem tensor from the left and the x tensor from the right so that the
        relative position encoding can be calculated correctly. For example:

        mem: 0, 1, 2, 3, 4
        x:   0, 1, 2, 3, 4, 5

        n_temporal = 1
        mem: 0, 1, 2, 3, 4   x: 0, 1, 2, 3, 4, 5

        n_temporal = 2
        mem: pad, 1, 3       x: 0, 2, 4
        mem: 0, 2, 4         x: 1, 3, 5

        n_temporal = 3
        mem: pad, 2          dec: 0, 3
        mem: 0,   3          dec: 1, 4
        mem: 1,   4          dec: 2, 5

        Args:
            x (paddle.Tensor): The input of the layer. It will have shape (B, T, H, W, C)
            mem (paddle.Tensor): The memory. It will have shape (B, T_mem, H, W, C)
            mem_global_vectors (paddle.Tensor): The global vectors from the memory. It will have shape (B, N, C)

        Returns:
            out (paddle.Tensor): Output tensor should have shape (B, T, H, W, C_out)
        """

        if self.cross_last_n_frames is not None:
            cross_last_n_frames = int(min(self.cross_last_n_frames, mem.shape[1]))
            mem = mem[:, -cross_last_n_frames:, ...]
        if self.use_global_vector:
            _, num_global, _ = mem_global_vectors.shape
        x = self.norm(x)
        B, T_x, H, W, C_in = x.shape
        B_mem, T_mem, H_mem, W_mem, C_mem = mem.shape
        assert T_x < self.max_temporal_relative and T_mem < self.max_temporal_relative
        cuboid_hw = self.cuboid_hw
        n_temporal = self.n_temporal
        shift_hw = self.shift_hw
        assert (
            B_mem == B and H == H_mem and W == W_mem and C_in == C_mem
        ), f"Shape of memory and the input tensor does not match. x.shape={x.shape}, mem.shape={mem.shape}"
        pad_t_mem = (n_temporal - T_mem % n_temporal) % n_temporal
        pad_t_x = (n_temporal - T_x % n_temporal) % n_temporal
        pad_h = (cuboid_hw[0] - H % cuboid_hw[0]) % cuboid_hw[0]
        pad_w = (cuboid_hw[1] - W % cuboid_hw[1]) % cuboid_hw[1]
        mem = cuboid_utils.generalize_padding(
            mem, pad_t_mem, pad_h, pad_w, self.padding_type, t_pad_left=True
        )

        x = cuboid_utils.generalize_padding(
            x, pad_t_x, pad_h, pad_w, self.padding_type, t_pad_left=False
        )

        if any(i > 0 for i in shift_hw):
            shifted_x = paddle.roll(
                x=x, shifts=(-shift_hw[0], -shift_hw[1]), axis=(2, 3)
            )
            shifted_mem = paddle.roll(
                x=mem, shifts=(-shift_hw[0], -shift_hw[1]), axis=(2, 3)
            )
        else:
            shifted_x = x
            shifted_mem = mem
        mem_cuboid_size = (mem.shape[1] // n_temporal,) + cuboid_hw
        x_cuboid_size = (x.shape[1] // n_temporal,) + cuboid_hw
        reordered_mem = cuboid_encoder.cuboid_reorder(
            shifted_mem, cuboid_size=mem_cuboid_size, strategy=self.strategy
        )
        reordered_x = cuboid_encoder.cuboid_reorder(
            shifted_x, cuboid_size=x_cuboid_size, strategy=self.strategy
        )
        _, num_cuboids_mem, mem_cuboid_volume, _ = reordered_mem.shape
        _, num_cuboids, x_cuboid_volume, _ = reordered_x.shape
        assert (
            num_cuboids_mem == num_cuboids
        ), f"Number of cuboids do not match. num_cuboids={num_cuboids}, num_cuboids_mem={num_cuboids_mem}"
        attn_mask = compute_cuboid_cross_attention_mask(
            T_x,
            T_mem,
            H,
            W,
            n_temporal,
            cuboid_hw,
            shift_hw,
            strategy=self.strategy,
            padding_type=self.padding_type,
            device=x.place,
        )
        head_C = C_in // self.num_heads
        kv = (
            self.kv_proj(reordered_mem)
            .reshape([B, num_cuboids, mem_cuboid_volume, 2, self.num_heads, head_C])
            .transpose(perm=[3, 0, 4, 1, 2, 5])
        )
        k, v = kv[0], kv[1]
        q = (
            self.q_proj(reordered_x)
            .reshape([B, num_cuboids, x_cuboid_volume, self.num_heads, head_C])
            .transpose(perm=[0, 3, 1, 2, 4])
        )
        q = q * self.scale
        perm_4 = list(range(k.ndim))
        perm_4[-2] = -1
        perm_4[-1] = -2
        attn_score = q @ k.transpose(perm=perm_4)
        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[
                    :x_cuboid_volume, :mem_cuboid_volume
                ].reshape([-1])
            ].reshape([x_cuboid_volume, mem_cuboid_volume, -1])
            relative_position_bias = relative_position_bias.transpose(
                perm=[2, 0, 1]
            ).unsqueeze(axis=1)
            attn_score = attn_score + relative_position_bias
        if self.use_global_vector:
            if self.separate_global_qkv:
                l2g_q = (
                    self.l2g_q_net(reordered_x)
                    .reshape([B, num_cuboids, x_cuboid_volume, self.num_heads, head_C])
                    .transpose(perm=[0, 3, 1, 2, 4])
                )
                l2g_q = l2g_q * self.scale
                l2g_global_kv = (
                    self.l2g_global_kv_net(mem_global_vectors)
                    .reshape([B, 1, num_global, 2, self.num_heads, head_C])
                    .transpose(perm=[3, 0, 4, 1, 2, 5])
                )
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
            else:
                kv_global = (
                    self.kv_proj(mem_global_vectors)
                    .reshape([B, 1, num_global, 2, self.num_heads, head_C])
                    .transpose(perm=[3, 0, 4, 1, 2, 5])
                )
                l2g_global_k, l2g_global_v = kv_global[0], kv_global[1]
                l2g_q = q
            perm_5 = list(range(l2g_global_k.ndim))
            perm_5[-2] = -1
            perm_5[-1] = -2
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(perm=perm_5)
            attn_score_l2l_l2g = paddle.concat(x=(attn_score, l2g_attn_score), axis=-1)
            if attn_mask.ndim == 5:
                attn_mask_l2l_l2g = F.pad(
                    attn_mask, [0, num_global], "constant", 1, data_format="NDHWC"
                )
            else:
                attn_mask_l2l_l2g = F.pad(attn_mask, [0, num_global], "constant", 1)
            v_l_g = paddle.concat(
                x=(
                    v,
                    l2g_global_v.expand(
                        shape=[B, self.num_heads, num_cuboids, num_global, head_C]
                    ),
                ),
                axis=3,
            )
            attn_score_l2l_l2g = cuboid_encoder.masked_softmax(
                attn_score_l2l_l2g, mask=attn_mask_l2l_l2g
            )
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)
            reordered_x = (
                (attn_score_l2l_l2g @ v_l_g)
                .transpose(perm=[0, 2, 3, 1, 4])
                .reshape(B, num_cuboids, x_cuboid_volume, self.dim)
            )
        else:
            attn_score = cuboid_encoder.masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)
            reordered_x = (
                (attn_score @ v)
                .transpose(perm=[0, 2, 3, 1, 4])
                .reshape([B, num_cuboids, x_cuboid_volume, self.dim])
            )
        reordered_x = paddle.cast(reordered_x, dtype="float32")
        reordered_x = self.proj_drop(self.proj(reordered_x))
        shifted_x = cuboid_encoder.cuboid_reorder_reverse(
            reordered_x,
            cuboid_size=x_cuboid_size,
            strategy=self.strategy,
            orig_data_shape=(x.shape[1], x.shape[2], x.shape[3]),
        )
        if any(i > 0 for i in shift_hw):
            x = paddle.roll(x=shifted_x, shifts=(shift_hw[0], shift_hw[1]), axis=(2, 3))
        else:
            x = shifted_x
        x = cuboid_utils.generalize_unpadding(
            x, pad_t=pad_t_x, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type
        )
        return x


class StackCuboidCrossAttentionBlock(paddle.nn.Layer):
    """A stack of cuboid cross attention layers.

    The advantage of cuboid attention is that we can combine cuboid attention building blocks with different
    hyper-parameters to mimic a broad range of space-time correlation patterns.

    - "use_inter_ffn" is True
        x, mem --> attn1 -----+-------> ffn1 ---+---> attn2 --> ... --> ffn_k --> out
           |             ^    |             ^
           |             |    |             |
           |-------------|----|-------------|
    - "use_inter_ffn" is False
        x, mem --> attn1 -----+------> attn2 --> ... attnk --+----> ffnk ---+---> out, mem
           |             ^    |            ^             ^  |           ^
           |             |    |            |             |  |           |
           |-------------|----|------------|-- ----------|--|-----------|

    Args:
        dim (int): The dimension of the input.
        num_heads (int): The number of head.
        block_cuboid_hw (list, optional): The height and width of block cuboid.Defaults to [(4, 4), (4, 4)].
        block_shift_hw (list, optional): The height and width of shift cuboid . Defaults to [(0, 0), (2, 2)].
        block_n_temporal (list, optional): The length of block temporal. Defaults to [1, 2].
        block_strategy (list, optional): The strategy of block. Defaults to [("d", "d", "d"), ("l", "l", "l")].
        padding_type (str, optional): The type of paddling. Defaults to "ignore".
        cross_last_n_frames (int, optional): The num of cross_last_n_frames. Defaults to None.
        qkv_bias (bool, optional): Whether to enable bias in calculating qkv attention. Defaults to False.
        qk_scale (float, optional): Whether to enable scale factor when calculating the attention. Defaults to None.
        attn_drop (float, optional): The attention dropout. Defaults to 0.0.
        proj_drop (float, optional): The projection dropout. Defaults to 0.0.
        ffn_drop (float, optional): The ratio of FFN dropout. Defaults to 0.0.
        activation (str, optional): The activation. Defaults to "leaky".
        gated_ffn (bool, optional): Whether to use gate FFN. Defaults to False.
        norm_layer (str, optional): The normalization layer. Defaults to "layer_norm".
        use_inter_ffn (bool, optional): Whether to use inter FFN. Defaults to True.
        max_temporal_relative (int, optional): The max temporal. Defaults to 50.
        checkpoint_level (int, optional): Whether to enable gradient checkpointing. Defaults to 1.
        use_relative_pos (bool, optional): Whether to use relative pos. Defaults to True.
        use_global_vector (bool, optional): Whether to use the global vector or not. Defaults to False.
        separate_global_qkv (bool, optional): Whether to use different network to calc q_global, k_global, v_global. Defaults to False.
        global_dim_ratio (int, optional): The dim (channels) of global vectors is `global_dim_ratio*dim`. Defaults to 1.
        attn_linear_init_mode (str, optional): The mode of attention linear initialization. Defaults to "0".
        ffn_linear_init_mode (str, optional): The mode of FFN linear initialization. Defaults to "0".
        norm_init_mode (str, optional): The mode of normalization. Defaults to "0".
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_cuboid_hw: Tuple[Tuple[int, ...], ...] = [(4, 4), (4, 4)],
        block_shift_hw: Tuple[Tuple[int, ...], ...] = [(0, 0), (2, 2)],
        block_n_temporal: Tuple[int, ...] = [1, 2],
        block_strategy: Tuple[Tuple[str, ...], ...] = [
            ("d", "d", "d"),
            ("l", "l", "l"),
        ],
        padding_type: str = "ignore",
        cross_last_n_frames: int = None,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        activation: str = "leaky",
        gated_ffn: bool = False,
        norm_layer: str = "layer_norm",
        use_inter_ffn: bool = True,
        max_temporal_relative: int = 50,
        checkpoint_level: int = 1,
        use_relative_pos: bool = True,
        use_global_vector: bool = False,
        separate_global_qkv: bool = False,
        global_dim_ratio: int = 1,
        attn_linear_init_mode: str = "0",
        ffn_linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(StackCuboidCrossAttentionBlock, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode
        if (
            len(block_cuboid_hw[0]) <= 0
            or len(block_shift_hw) <= 0
            or len(block_strategy) <= 0
        ):
            raise ValueError(
                "Incorrect format.The lengths of block_cuboid_hw[0], block_shift_hw, and block_strategy must be greater than zero."
            )
        if len(block_cuboid_hw) != len(block_shift_hw) and len(block_shift_hw) == len(
            block_strategy
        ):
            raise ValueError(
                "The lengths of block_cuboid_size, block_shift_size, and block_strategy must be equal."
            )

        self.num_attn = len(block_cuboid_hw)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        self.use_global_vector = use_global_vector
        if self.use_inter_ffn:
            self.ffn_l = paddle.nn.LayerList(
                sublayers=[
                    cuboid_encoder.PositionwiseFFN(
                        units=dim,
                        hidden_size=4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn,
                        activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        norm_init_mode=norm_init_mode,
                    )
                    for _ in range(self.num_attn)
                ]
            )
        else:
            self.ffn_l = paddle.nn.LayerList(
                sublayers=[
                    cuboid_encoder.PositionwiseFFN(
                        units=dim,
                        hidden_size=4 * dim,
                        activation_dropout=ffn_drop,
                        dropout=ffn_drop,
                        gated_proj=gated_ffn,
                        activation=activation,
                        normalization=norm_layer,
                        pre_norm=True,
                        linear_init_mode=ffn_linear_init_mode,
                        norm_init_mode=norm_init_mode,
                    )
                ]
            )
        self.attn_l = paddle.nn.LayerList(
            sublayers=[
                CuboidCrossAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    cuboid_hw=ele_cuboid_hw,
                    shift_hw=ele_shift_hw,
                    strategy=ele_strategy,
                    n_temporal=ele_n_temporal,
                    cross_last_n_frames=cross_last_n_frames,
                    padding_type=padding_type,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    max_temporal_relative=max_temporal_relative,
                    use_global_vector=use_global_vector,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for ele_cuboid_hw, ele_shift_hw, ele_strategy, ele_n_temporal in zip(
                    block_cuboid_hw, block_shift_hw, block_strategy, block_n_temporal
                )
            ]
        )

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, mem, mem_global_vector=None):
        """
        Args:
            x (paddle.Tensor): Shape (B, T_x, H, W, C)
            mem (paddle.Tensor): Shape (B, T_mem, H, W, C)
            mem_global_vector (paddle.Tensor): Shape (B, N_global, C)

        Returns:
            out (paddle.Tensor): (B, T_x, H, W, C_out)
        """

        if self.use_inter_ffn:
            for attn, ffn in zip(self.attn_l, self.ffn_l):
                if self.checkpoint_level >= 2 and self.training:
                    x = x + fleet.utils.recompute(attn, x, mem, mem_global_vector)
                else:
                    x = x + attn(x, mem, mem_global_vector)
                if self.checkpoint_level >= 1 and self.training:
                    x = fleet.utils.recompute(ffn, x)
                else:
                    x = ffn(x)
            return x
        else:
            for attn in self.attn_l:
                if self.checkpoint_level >= 2 and self.training:
                    x = x + fleet.utils.recompute(attn, x, mem, mem_global_vector)
                else:
                    x = x + attn(x, mem, mem_global_vector)
            if self.checkpoint_level >= 1 and self.training:
                x = fleet.utils.recompute(self.ffn_l[0], x)
            else:
                x = self.ffn_l[0](x)
        return x


class Upsample3DLayer(paddle.nn.Layer):
    """Upsampling based on nn.UpSampling and Conv3x3.

    If the temporal dimension remains the same:
        x --> interpolation-2d (nearest) --> conv3x3(dim, out_dim)
    Else:
        x --> interpolation-3d (nearest) --> conv3x3x3(dim, out_dim)

    Args:
        dim (int): The dimension of the input tensor.
        out_dim (int): The dimension of the output tensor.
        target_size (Tuple[int,...]): The size of output tensor.
        temporal_upsample (bool, optional): Whether the temporal axis will go through upsampling. Defaults to False.
        kernel_size (int, optional): The kernel size of the Conv2D layer. Defaults to 3.
        layout (str, optional): The layout of the inputs. Defaults to "THWC".
        conv_init_mode (str, optional): The mode of conv initialization. Defaults to "0".
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        target_size: Tuple[int, ...],
        temporal_upsample: bool = False,
        kernel_size: int = 3,
        layout: str = "THWC",
        conv_init_mode: str = "0",
    ):
        super(Upsample3DLayer, self).__init__()
        self.conv_init_mode = conv_init_mode
        self.target_size = target_size
        self.out_dim = out_dim
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up = paddle.nn.Upsample(size=target_size, mode="nearest")
        else:
            self.up = paddle.nn.Upsample(
                size=(target_size[1], target_size[2]), mode="nearest"
            )
        self.conv = paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=out_dim,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2),
        )
        assert layout in ["THWC", "CTHW"]
        self.layout = layout
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(m, conv_mode=self.conv_init_mode)

    def forward(self, x):
        """

        Args:
            x : (B, T, H, W, C) or (B, C, T, H, W)

        Returns:
            out : (B, T, H_new, W_out, C_out) or (B, C, T, H_out, W_out)
        """

        if self.layout == "THWC":
            B, T, H, W, C = x.shape
            if self.temporal_upsample:
                x = x.transpose(perm=[0, 4, 1, 2, 3])
                return self.conv(self.up(x)).transpose(perm=[0, 2, 3, 4, 1])
            else:
                assert self.target_size[0] == T
                x = x.reshape([B * T, H, W, C]).transpose(perm=[0, 3, 1, 2])
                x = self.up(x)
                return (
                    self.conv(x)
                    .transpose(perm=[0, 2, 3, 1])
                    .reshape(list((B,) + self.target_size + (self.out_dim,)))
                )
        elif self.layout == "CTHW":
            B, C, T, H, W = x.shape
            if self.temporal_upsample:
                return self.conv(self.up(x))
            else:
                assert self.output_size[0] == T
                x = x.transpose(perm=[0, 2, 1, 3, 4])
                x = x.reshape([B * T, C, H, W])
                return (
                    self.conv(self.up(x))
                    .reshape(
                        [
                            B,
                            self.target_size[0],
                            self.out_dim,
                            self.target_size[1],
                            self.target_size[2],
                        ]
                    )
                    .transpose(perm=[0, 2, 1, 3, 4])
                )


class CuboidTransformerDecoder(paddle.nn.Layer):
    """Decoder of the CuboidTransformer.

    For each block, we first apply the StackCuboidSelfAttention and then apply the StackCuboidCrossAttention

    Repeat the following structure K times

        x --> StackCuboidSelfAttention --> |
                                           |----> StackCuboidCrossAttention (If used) --> out
                                   mem --> |

    Args:
        target_temporal_length (int): The temporal length of the target.
        mem_shapes (Tuple[int,...]): The mem shapes of the decoder.
        cross_start (int, optional): The block to start cross attention. Defaults to 0.
        depth (list, optional): The number of layers for each block. Defaults to [2, 2].
        upsample_type (str, optional): The type of upsample. Defaults to "upsample".
        upsample_kernel_size (int, optional): The kernel size of upsample. Defaults to 3.
        block_self_attn_patterns (str, optional): The patterns of block attention. Defaults to None.
        block_self_cuboid_size (list, optional): The size of cuboid block. Defaults to [(4, 4, 4), (4, 4, 4)].
        block_self_cuboid_strategy (list, optional): The strategy of cuboid. Defaults to [("l", "l", "l"), ("d", "d", "d")].
        block_self_shift_size (list, optional): The size of shift. Defaults to [(1, 1, 1), (0, 0, 0)].
        block_cross_attn_patterns (str, optional): The patterns of cross attentions. Defaults to None.
        block_cross_cuboid_hw (list, optional): The height and width of  cross cuboid. Defaults to [(4, 4), (4, 4)].
        block_cross_cuboid_strategy (list, optional): The strategy of cross cuboid. Defaults to [("l", "l", "l"), ("d", "l", "l")].
        block_cross_shift_hw (list, optional): The height and width of  cross shift. Defaults to [(0, 0), (0, 0)].
        block_cross_n_temporal (list, optional): The cross temporal of block. Defaults to [1, 2].
        cross_last_n_frames (int, optional): The num of cross last frames. Defaults to None.
        num_heads (int, optional): The num of head. Defaults to 4.
        attn_drop (float, optional): The ratio of attention dropout. Defaults to 0.0.
        proj_drop (float, optional): The ratio of projection dropout. Defaults to 0.0.
        ffn_drop (float, optional): The ratio of FFN dropout. Defaults to 0.0.
        ffn_activation (str, optional): The activation layer of FFN. Defaults to "leaky".
        gated_ffn (bool, optional): Whether to use gate FFN. Defaults to False.
        norm_layer (str, optional): The normalization layer. Defaults to "layer_norm".
        use_inter_ffn (bool, optional): Whether to use inter FFN. Defaults to False.
        hierarchical_pos_embed (bool, optional): Whether to use hierarchical pos_embed. Defaults to False.
        pos_embed_type (str, optional): The type of pos embeding. Defaults to "t+hw".
        max_temporal_relative (int, optional): The max number of teemporal relative. Defaults to 50.
        padding_type (str, optional): The type of padding. Defaults to "ignore".
        checkpoint_level (bool, optional):  Whether to enable gradient checkpointing. Defaults to True.
        use_relative_pos (bool, optional):  Whether to use relative pos. Defaults to True.
        self_attn_use_final_proj (bool, optional): Whether to use self attention for final projection. Defaults to True.
        use_first_self_attn (bool, optional): Whether to use first self attention. Defaults to False.
        use_self_global (bool, optional): Whether to use self global vector. Defaults to False.
        self_update_global (bool, optional): Whether to update global vector. Defaults to True.
        use_cross_global (bool, optional):  Whether to use cross global vector. Defaults to False.
        use_global_vector_ffn (bool, optional): Whether to use FFN global vectors. Defaults to True.
        use_global_self_attn (bool, optional):  Whether to use global self attention. Defaults to False.
        separate_global_qkv (bool, optional): Whether to use different network to calc q_global, k_global, v_global. Defaults to False.
        global_dim_ratio (int, optional): The dim (channels) of global vectors is `global_dim_ratio*dim`. Defaults to 1.
        attn_linear_init_mode (str, optional): The mode of attention linear initialization. Defaults to "0".
        ffn_linear_init_mode (str, optional): The mode of FFN linear initialization. Defaults to "0".
        conv_init_mode (str, optional): The mode of conv initialization. Defaults to "0".
        up_linear_init_mode (str, optional): The mode of up linear initialization. Defaults to "0".
        norm_init_mode (str, optional): The mode of normalization initialization. Defaults to "0".
    """

    def __init__(
        self,
        target_temporal_length: int,
        mem_shapes: Tuple[int, ...],
        cross_start: int = 0,
        depth: Tuple[int, ...] = [2, 2],
        upsample_type: str = "upsample",
        upsample_kernel_size: int = 3,
        block_self_attn_patterns: str = None,
        block_self_cuboid_size: Tuple[Tuple[int, ...], ...] = [(4, 4, 4), (4, 4, 4)],
        block_self_cuboid_strategy: Tuple[Tuple[str, ...], ...] = [
            ("l", "l", "l"),
            ("d", "d", "d"),
        ],
        block_self_shift_size: Tuple[Tuple[int, ...], ...] = [(1, 1, 1), (0, 0, 0)],
        block_cross_attn_patterns: str = None,
        block_cross_cuboid_hw: Tuple[Tuple[int, ...], ...] = [(4, 4), (4, 4)],
        block_cross_cuboid_strategy: Tuple[Tuple[str, ...], ...] = [
            ("l", "l", "l"),
            ("d", "l", "l"),
        ],
        block_cross_shift_hw: Tuple[Tuple[int, ...], ...] = [(0, 0), (0, 0)],
        block_cross_n_temporal: Tuple[int, ...] = [1, 2],
        cross_last_n_frames: int = None,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        ffn_activation: str = "leaky",
        gated_ffn: bool = False,
        norm_layer: str = "layer_norm",
        use_inter_ffn: bool = False,
        hierarchical_pos_embed: bool = False,
        pos_embed_type: str = "t+hw",
        max_temporal_relative: int = 50,
        padding_type: str = "ignore",
        checkpoint_level: bool = True,
        use_relative_pos: bool = True,
        self_attn_use_final_proj: bool = True,
        use_first_self_attn: bool = False,
        use_self_global: bool = False,
        self_update_global: bool = True,
        use_cross_global: bool = False,
        use_global_vector_ffn: bool = True,
        use_global_self_attn: bool = False,
        separate_global_qkv: bool = False,
        global_dim_ratio: int = 1,
        attn_linear_init_mode: str = "0",
        ffn_linear_init_mode: str = "0",
        conv_init_mode: str = "0",
        up_linear_init_mode: str = "0",
        norm_init_mode: str = "0",
    ):
        super(CuboidTransformerDecoder, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.up_linear_init_mode = up_linear_init_mode
        self.norm_init_mode = norm_init_mode
        assert len(depth) == len(mem_shapes)
        self.target_temporal_length = target_temporal_length
        self.num_blocks = len(mem_shapes)
        self.cross_start = cross_start
        self.mem_shapes = mem_shapes
        self.depth = depth
        self.upsample_type = upsample_type
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.checkpoint_level = checkpoint_level
        self.use_self_global = use_self_global
        self.self_update_global = self_update_global
        self.use_cross_global = use_cross_global
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_first_self_attn = use_first_self_attn
        if block_self_attn_patterns is not None:
            if isinstance(block_self_attn_patterns, (tuple, list)):
                assert len(block_self_attn_patterns) == self.num_blocks
            else:
                block_self_attn_patterns = [
                    block_self_attn_patterns for _ in range(self.num_blocks)
                ]
            block_self_cuboid_size = []
            block_self_cuboid_strategy = []
            block_self_shift_size = []
            for idx, key in enumerate(block_self_attn_patterns):
                func = cuboid_utils.CuboidSelfAttentionPatterns.get(key)
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_self_cuboid_size.append(cuboid_size)
                block_self_cuboid_strategy.append(strategy)
                block_self_shift_size.append(shift_size)
        else:
            if not isinstance(block_self_cuboid_size[0][0], (list, tuple)):
                block_self_cuboid_size = [
                    block_self_cuboid_size for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_self_cuboid_size) == self.num_blocks
                ), f"Incorrect input format! Received block_self_cuboid_size={block_self_cuboid_size}"
            if not isinstance(block_self_cuboid_strategy[0][0], (list, tuple)):
                block_self_cuboid_strategy = [
                    block_self_cuboid_strategy for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_self_cuboid_strategy) == self.num_blocks
                ), f"Incorrect input format! Received block_self_cuboid_strategy={block_self_cuboid_strategy}"
            if not isinstance(block_self_shift_size[0][0], (list, tuple)):
                block_self_shift_size = [
                    block_self_shift_size for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_self_shift_size) == self.num_blocks
                ), f"Incorrect input format! Received block_self_shift_size={block_self_shift_size}"
        self_blocks = []
        for i in range(self.num_blocks):
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                ele_depth = depth[i] - 1
            else:
                ele_depth = depth[i]
            stack_cuboid_blocks = [
                cuboid_encoder.StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_self_cuboid_size[i],
                    block_strategy=block_self_cuboid_strategy[i],
                    block_shift_size=block_self_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_self_global,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for _ in range(ele_depth)
            ]
            self_blocks.append(paddle.nn.LayerList(sublayers=stack_cuboid_blocks))
        self.self_blocks = paddle.nn.LayerList(sublayers=self_blocks)
        if block_cross_attn_patterns is not None:
            if isinstance(block_cross_attn_patterns, (tuple, list)):
                assert len(block_cross_attn_patterns) == self.num_blocks
            else:
                block_cross_attn_patterns = [
                    block_cross_attn_patterns for _ in range(self.num_blocks)
                ]
            block_cross_cuboid_hw = []
            block_cross_cuboid_strategy = []
            block_cross_shift_hw = []
            block_cross_n_temporal = []
            for idx, key in enumerate(block_cross_attn_patterns):
                if key == "last_frame_dst":
                    cuboid_hw = None
                    shift_hw = None
                    strategy = None
                    n_temporal = None
                else:
                    func = cuboid_utils.CuboidCrossAttentionPatterns.get(key)
                    cuboid_hw, shift_hw, strategy, n_temporal = func(mem_shapes[idx])
                block_cross_cuboid_hw.append(cuboid_hw)
                block_cross_cuboid_strategy.append(strategy)
                block_cross_shift_hw.append(shift_hw)
                block_cross_n_temporal.append(n_temporal)
        else:
            if not isinstance(block_cross_cuboid_hw[0][0], (list, tuple)):
                block_cross_cuboid_hw = [
                    block_cross_cuboid_hw for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_cross_cuboid_hw) == self.num_blocks
                ), f"Incorrect input format! Received block_cross_cuboid_hw={block_cross_cuboid_hw}"
            if not isinstance(block_cross_cuboid_strategy[0][0], (list, tuple)):
                block_cross_cuboid_strategy = [
                    block_cross_cuboid_strategy for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_cross_cuboid_strategy) == self.num_blocks
                ), f"Incorrect input format! Received block_cross_cuboid_strategy={block_cross_cuboid_strategy}"
            if not isinstance(block_cross_shift_hw[0][0], (list, tuple)):
                block_cross_shift_hw = [
                    block_cross_shift_hw for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_cross_shift_hw) == self.num_blocks
                ), f"Incorrect input format! Received block_cross_shift_hw={block_cross_shift_hw}"
            if not isinstance(block_cross_n_temporal[0], (list, tuple)):
                block_cross_n_temporal = [
                    block_cross_n_temporal for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_cross_n_temporal) == self.num_blocks
                ), f"Incorrect input format! Received block_cross_n_temporal={block_cross_n_temporal}"
        self.cross_blocks = paddle.nn.LayerList()
        for i in range(self.cross_start, self.num_blocks):
            cross_block = paddle.nn.LayerList(
                sublayers=[
                    StackCuboidCrossAttentionBlock(
                        dim=self.mem_shapes[i][-1],
                        num_heads=num_heads,
                        block_cuboid_hw=block_cross_cuboid_hw[i],
                        block_strategy=block_cross_cuboid_strategy[i],
                        block_shift_hw=block_cross_shift_hw[i],
                        block_n_temporal=block_cross_n_temporal[i],
                        cross_last_n_frames=cross_last_n_frames,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        ffn_drop=ffn_drop,
                        gated_ffn=gated_ffn,
                        norm_layer=norm_layer,
                        use_inter_ffn=use_inter_ffn,
                        activation=ffn_activation,
                        max_temporal_relative=max_temporal_relative,
                        padding_type=padding_type,
                        use_global_vector=use_cross_global,
                        separate_global_qkv=separate_global_qkv,
                        global_dim_ratio=global_dim_ratio,
                        checkpoint_level=checkpoint_level,
                        use_relative_pos=use_relative_pos,
                        attn_linear_init_mode=attn_linear_init_mode,
                        ffn_linear_init_mode=ffn_linear_init_mode,
                        norm_init_mode=norm_init_mode,
                    )
                    for _ in range(depth[i])
                ]
            )
            self.cross_blocks.append(cross_block)
        if self.num_blocks > 1:
            if self.upsample_type == "upsample":
                self.upsample_layers = paddle.nn.LayerList(
                    sublayers=[
                        Upsample3DLayer(
                            dim=self.mem_shapes[i + 1][-1],
                            out_dim=self.mem_shapes[i][-1],
                            target_size=(target_temporal_length,)
                            + self.mem_shapes[i][1:3],
                            kernel_size=upsample_kernel_size,
                            temporal_upsample=False,
                            conv_init_mode=conv_init_mode,
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            else:
                raise NotImplementedError(f"{self.upsample_type} is invalid.")
            if self.hierarchical_pos_embed:
                self.hierarchical_pos_embed_l = paddle.nn.LayerList(
                    sublayers=[
                        PosEmbed(
                            embed_dim=self.mem_shapes[i][-1],
                            typ=pos_embed_type,
                            maxT=target_temporal_length,
                            maxH=self.mem_shapes[i][1],
                            maxW=self.mem_shapes[i][2],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
        self.reset_parameters()

    def reset_parameters(self):
        for ms in self.self_blocks:
            for m in ms:
                m.reset_parameters()
        for ms in self.cross_blocks:
            for m in ms:
                m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.upsample_layers:
                m.reset_parameters()
        if self.hierarchical_pos_embed:
            for m in self.hierarchical_pos_embed_l:
                m.reset_parameters()

    def forward(self, x, mem_l, mem_global_vector_l=None):
        """
        Args:
            x : Shape (B, T_top, H_top, W_top, C).
            mem_l : A list of memory tensors.
        """

        B, T_top, H_top, W_top, C = x.shape
        assert T_top == self.target_temporal_length
        assert (H_top, W_top) == (self.mem_shapes[-1][1], self.mem_shapes[-1][2])
        for i in range(self.num_blocks - 1, -1, -1):
            mem_global_vector = (
                None if mem_global_vector_l is None else mem_global_vector_l[i]
            )
            if not self.use_first_self_attn and i == self.num_blocks - 1:
                if i >= self.cross_start:
                    x = self.cross_blocks[i - self.cross_start][0](
                        x, mem_l[i], mem_global_vector
                    )
                for idx in range(self.depth[i] - 1):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](
                                x, mem_global_vector
                            )
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx + 1](
                            x, mem_l[i], mem_global_vector
                        )
            else:
                for idx in range(self.depth[i]):
                    if self.use_self_global:
                        if self.self_update_global:
                            x, mem_global_vector = self.self_blocks[i][idx](
                                x, mem_global_vector
                            )
                        else:
                            x, _ = self.self_blocks[i][idx](x, mem_global_vector)
                    else:
                        x = self.self_blocks[i][idx](x)
                    if i >= self.cross_start:
                        x = self.cross_blocks[i - self.cross_start][idx](
                            x, mem_l[i], mem_global_vector
                        )
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.hierarchical_pos_embed_l[i - 1](x)
        return x

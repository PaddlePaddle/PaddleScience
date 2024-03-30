import numpy as np
import paddle
from paddle import nn
from collections import OrderedDict
import paddle.nn.functional as F
from functools import lru_cache
from paddle.distributed import fleet

from ppsci.utils import initializer
from ppsci.arch import activation as act_mod
import ppsci.arch.cuboid_transformer_utils as cuboid_utils

NEGATIVE_SLOPE = 0.1

class PatchMerging3D(paddle.nn.Layer):
    """Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        downsample (Tuple[int,]): downsample factor
        norm_layer :The normalization layer

    """

    def __init__(
        self,
        dim,
        out_dim=None,
        downsample=(1, 2, 2),
        norm_layer="layer_norm",
        padding_type="nearest",
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.dim = dim
        if out_dim is None:
            out_dim = max(downsample) * dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.padding_type = padding_type
        self.reduction = paddle.nn.Linear(
            in_features=downsample[0] * downsample[1] * downsample[2] * dim,
            out_features=out_dim,
            bias_attr=False,
        )
        self.norm = cuboid_utils.get_norm_layer(
            norm_layer, in_channels=downsample[0] * downsample[1] * downsample[2] * dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.children():
            cuboid_utils.apply_initialization(
                m, linear_mode=self.linear_init_mode, norm_mode=self.norm_init_mode
            )

    def get_out_shape(self, data_shape):
        T, H, W, C_in = data_shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        return (
            (T + pad_t) // self.downsample[0],
            (H + pad_h) // self.downsample[1],
            (W + pad_w) // self.downsample[2],
            self.out_dim,
        )

    def forward(self, x):
        """

        Args:
            x : (B, T, H, W, C)

        Returns:
            out : Shape (B, T // downsample[0], H // downsample[1], W // downsample[2], out_dim)
        """

        B, T, H, W, C = x.shape
        pad_t = (self.downsample[0] - T % self.downsample[0]) % self.downsample[0]
        pad_h = (self.downsample[1] - H % self.downsample[1]) % self.downsample[1]
        pad_w = (self.downsample[2] - W % self.downsample[2]) % self.downsample[2]
        if pad_h or pad_h or pad_w:
            T += pad_t
            H += pad_h
            W += pad_w
            x = cuboid_utils.generalize_padding(
                x, pad_t, pad_h, pad_w, padding_type=self.padding_type
            )
        x = (
            x.reshape(
                (
                    B,
                    T // self.downsample[0],
                    self.downsample[0],
                    H // self.downsample[1],
                    self.downsample[1],
                    W // self.downsample[2],
                    self.downsample[2],
                    C,
                )
            )
            .transpose(perm=[0, 1, 3, 5, 2, 4, 6, 7])
            .reshape(
                [
                    B,
                    T // self.downsample[0],
                    H // self.downsample[1],
                    W // self.downsample[2],
                    self.downsample[0] * self.downsample[1] * self.downsample[2] * C,
                ]
            )
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PositionwiseFFN(paddle.nn.Layer):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data

    Args:
        units (int):
        hidden_size (int):
        activation_dropout (float):
        dropout (float):
        activation (str):
        normalization (str): layer_norm or no_norm
        layer_norm_eps (float): layer_norm epsilon
        pre_norm (bool): Pre-layer normalization as proposed in the paper:
                "[ACL2018] The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation" This will stabilize the training of Transformers.
                You may also refer to "[Arxiv2020] Understanding the Difficulty of Training Transformers"
    """

    def __init__(
        self,
        units: int = 512,
        hidden_size: int = 2048,
        activation_dropout: float = 0.0,
        dropout: float = 0.1,
        gated_proj: bool = False,
        activation="relu",
        normalization: str = "layer_norm",
        layer_norm_eps: float = 1e-05,
        pre_norm: bool = False,
        linear_init_mode="0",
        norm_init_mode="0",
    ):
        super().__init__()
        self.linear_init_mode = linear_init_mode
        self.norm_init_mode = norm_init_mode
        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict(
            [
                ("units", units),
                ("hidden_size", hidden_size),
                ("activation_dropout", activation_dropout),
                ("activation", activation),
                ("dropout", dropout),
                ("normalization", normalization),
                ("layer_norm_eps", layer_norm_eps),
                ("gated_proj", gated_proj),
                ("pre_norm", pre_norm),
            ]
        )
        self.dropout_layer = paddle.nn.Dropout(p=dropout)
        self.activation_dropout_layer = paddle.nn.Dropout(p=activation_dropout)
        self.ffn_1 = paddle.nn.Linear(
            in_features=units, out_features=hidden_size, bias_attr=True
        )
        if self._gated_proj:
            self.ffn_1_gate = paddle.nn.Linear(
                in_features=units, out_features=hidden_size, bias_attr=True
            )
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(NEGATIVE_SLOPE)
        else:
            self.activation = act_mod.get_activation(activation)
        self.ffn_2 = paddle.nn.Linear(
            in_features=hidden_size, out_features=units, bias_attr=True
        )
        self.layer_norm = cuboid_utils.get_norm_layer(
            normalization=normalization, in_channels=units, epsilon=layer_norm_eps
        )
        self.reset_parameters()

    def reset_parameters(self):
        cuboid_utils.apply_initialization(self.ffn_1, linear_mode=self.linear_init_mode)
        if self._gated_proj:
            cuboid_utils.apply_initialization(
                self.ffn_1_gate, linear_mode=self.linear_init_mode
            )
        cuboid_utils.apply_initialization(self.ffn_2, linear_mode=self.linear_init_mode)
        cuboid_utils.apply_initialization(
            self.layer_norm, norm_mode=self.norm_init_mode
        )

    def forward(self, data):
        """
        Args:
            x : Shape (B, seq_length, C_in)

        Returns:
            out : Shape (B, seq_length, C_out)
        """

        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out


def update_cuboid_size_shift_size(data_shape, cuboid_size, shift_size, strategy):
    """Update the cuboid_size and shift_size

    Args:
        data_shape : The shape of the data
        cuboid_size : Size of the cuboid
        shift_size : Size of the shift
        strategy : The strategy of attention

    Returns:
        new_cuboid_size : Size of the cuboid
        new_shift_size : Size of the shift
    """

    new_cuboid_size = list(cuboid_size)
    new_shift_size = list(shift_size)
    for i in range(len(data_shape)):
        if strategy[i] == "d":
            new_shift_size[i] = 0
        if data_shape[i] <= cuboid_size[i]:
            new_cuboid_size[i] = data_shape[i]
            new_shift_size[i] = 0
    return tuple(new_cuboid_size), tuple(new_shift_size)


def cuboid_reorder(data, cuboid_size, strategy):
    """Reorder the tensor into (B, num_cuboids, bT * bH * bW, C)
    We assume that the tensor shapes are divisible to the cuboid sizes.

    Args:
        data : The input data
        cuboid_size : The size of the cuboid
        strategy : The cuboid strategy

    Returns:
        reordered_data : Shape will be (B, num_cuboids, bT * bH * bW, C)
        num_cuboids = T / bT * H / bH * W / bW
    """

    B, T, H, W, C = data.shape
    num_cuboids = T // cuboid_size[0] * H // cuboid_size[1] * W // cuboid_size[2]
    cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
    intermediate_shape = []
    nblock_axis = []
    block_axis = []
    for i, (block_size, total_size, ele_strategy) in enumerate(
        zip(cuboid_size, (T, H, W), strategy)
    ):
        if ele_strategy == "l":
            intermediate_shape.extend([total_size // block_size, block_size])
            nblock_axis.append(2 * i + 1)
            block_axis.append(2 * i + 2)
        elif ele_strategy == "d":
            intermediate_shape.extend([block_size, total_size // block_size])
            nblock_axis.append(2 * i + 2)
            block_axis.append(2 * i + 1)
        else:
            raise NotImplementedError(f"{ele_strategy} is invalid.")
    data = data.reshape(list((B,) + tuple(intermediate_shape) + (C,)))
    reordered_data = data.transpose(
        perm=(0,) + tuple(nblock_axis) + tuple(block_axis) + (7,)
    )
    reordered_data = reordered_data.reshape((B, num_cuboids, cuboid_volume, C))
    return reordered_data


@lru_cache()
def compute_cuboid_self_attention_mask(
    data_shape, cuboid_size, shift_size, strategy, padding_type, device
):
    """Compute the shift window attention mask

    Args:
        data_shape : Should be T, H, W
        cuboid_size : Size of the cuboid
        shift_size : The shift size
        strategy : The decomposition strategy
        padding_type : Type of the padding
        device : The device

    Returns:
        attn_mask : Mask with shape (num_cuboid, cuboid_vol, cuboid_vol)
        The padded values will always be masked. The other masks will ensure that the shifted windows
        will only attend to those in the shifted windows.
    """
    T, H, W = data_shape
    pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
    pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
    pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
    data_mask = None
    if pad_t > 0 or pad_h > 0 or pad_w > 0:
        if padding_type == "ignore":
            data_mask = paddle.ones(shape=(1, T, H, W, 1), dtype="bool")
            data_mask = F.pad(
                data_mask, [0, 0, 0, pad_w, 0, pad_h, 0, pad_t], data_format="NDHWC"
            )
    else:
        data_mask = paddle.ones(
            shape=(1, T + pad_t, H + pad_h, W + pad_w, 1), dtype="bool"
        )
    if any(i > 0 for i in shift_size):
        if padding_type == "ignore":
            data_mask = paddle.roll(
                x=data_mask,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                axis=(1, 2, 3),
            )
    if padding_type == "ignore":
        data_mask = cuboid_reorder(data_mask, cuboid_size, strategy=strategy)
        data_mask = data_mask.squeeze(axis=-1).squeeze(axis=0)
    shift_mask = np.zeros(shape=(1, T + pad_t, H + pad_h, W + pad_w, 1))
    cnt = 0
    for t in (
        slice(-cuboid_size[0]),
        slice(-cuboid_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-cuboid_size[1]),
            slice(-cuboid_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-cuboid_size[2]),
                slice(-cuboid_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                shift_mask[:, t, h, w, :] = cnt
                cnt += 1
    shift_mask = paddle.to_tensor(shift_mask)
    shift_mask = cuboid_reorder(shift_mask, cuboid_size, strategy=strategy)
    shift_mask = shift_mask.squeeze(axis=-1).squeeze(axis=0)
    attn_mask = shift_mask.unsqueeze(axis=1) - shift_mask.unsqueeze(axis=2) == 0
    if padding_type == "ignore":
        attn_mask = (
            data_mask.unsqueeze(axis=1) * data_mask.unsqueeze(axis=2) * attn_mask
        )
    return attn_mask


def masked_softmax(att_score, mask, axis: int = -1):
    """Ignore the masked elements when calculating the softmax.
     The mask can be broadcastable.

    Args:
        att_score : Shape (..., length, ...)
        mask : Shape (..., length, ...)
            1 --> The element is not masked
            0 --> The element is masked
        axis : The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]

    Returns:
        att_weights : Shape (..., length, ...)
    """

    if mask is not None:
        if att_score.dtype == paddle.float16:
            att_score = att_score.masked_fill(paddle.logical_not(mask), -1e4)
        else:
            att_score = att_score.masked_fill(paddle.logical_not(mask), -1e18)
        att_weights = paddle.nn.functional.softmax(x=att_score, axis=axis) * mask
    else:
        att_weights = paddle.nn.functional.softmax(x=att_score, axis=axis)
    return att_weights


def cuboid_reorder_reverse(data, cuboid_size, strategy, orig_data_shape):
    """Reverse the reordered cuboid back to the original space

    Args:
        data :
        cuboid_size :
        strategy :
        orig_data_shape :

    Returns:
        data : The recovered data
    """

    B, num_cuboids, cuboid_volume, C = data.shape
    T, H, W = orig_data_shape
    permutation_axis = [0]
    for i, (block_size, total_size, ele_strategy) in enumerate(
        zip(cuboid_size, (T, H, W), strategy)
    ):
        if ele_strategy == "l":
            permutation_axis.append(i + 1)
            permutation_axis.append(i + 4)
        elif ele_strategy == "d":
            permutation_axis.append(i + 4)
            permutation_axis.append(i + 1)
        else:
            raise NotImplementedError((f"{ele_strategy} is invalid."))
    permutation_axis.append(7)
    data = data.reshape(
        [
            B,
            T // cuboid_size[0],
            H // cuboid_size[1],
            W // cuboid_size[2],
            cuboid_size[0],
            cuboid_size[1],
            cuboid_size[2],
            C,
        ]
    )
    data = data.transpose(perm=permutation_axis)
    data = data.reshape((B, T, H, W, C))
    return data


class CuboidSelfAttentionLayer(paddle.nn.Layer):
    """Implements the cuboid self attention.

    The idea of Cuboid Self Attention is to divide the input tensor (T, H, W) into several non-overlapping cuboids.
    We apply self-attention inside each cuboid and all cuboid-level self attentions are executed in parallel.

    We adopt two mechanisms for decomposing the input tensor into cuboids:

    1) local:
        We group the tensors within a local window, e.g., X[t:(t+b_t), h:(h+b_h), w:(w+b_w)]. We can also apply the
        shifted window strategy proposed in "[ICCV2021] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
    2) dilated:
        Inspired by the success of dilated convolution "[ICLR2016] Multi-Scale Context Aggregation by Dilated Convolutions",
         we split the tensor with dilation factors that are tied to the size of the cuboid. For example, for a cuboid that has width `b_w`,
         we sample the elements starting from 0 as 0, w / b_w, 2 * w / b_w, ..., (b_w - 1) * w / b_w.

    The cuboid attention can be viewed as a generalization of the attention mechanism proposed in Video Swin Transformer, https://arxiv.org/abs/2106.13230.
    The computational complexity of CuboidAttention can be simply calculated as O(T H W * b_t b_h b_w). To cover multiple correlation patterns,
    we are able to combine multiple CuboidAttention layers with different configurations such as cuboid size, shift size, and local / global decomposing strategy.

    In addition, it is straight-forward to extend the cuboid attention to other types of spatiotemporal data that are not described
    as regular tensors. We need to define alternative approaches to partition the data into "cuboids".

    In addition, inspired by "[NeurIPS2021] Do Transformers Really Perform Badly for Graph Representation?",
     "[NeurIPS2020] Big Bird: Transformers for Longer Sequences", "[EMNLP2021] Longformer: The Long-Document Transformer", we keep
     $K$ global vectors to record the global status of the spatiotemporal system. These global vectors will attend to the whole tensor and
     the vectors inside each individual cuboids will also attend to the global vectors so that they can peep into the global status of the system.

    Args:
        dim : The dimension of the input tensor
        num_heads : The number of heads
        cuboid_size : The size of each cuboid
        shift_size : The size for shifting the windows.
        strategy : The decomposition strategy of the tensor. 'l' stands for local and 'd' stands for dilated.
        padding_type : The type of padding.
        qkv_bias : Whether to enable bias in calculating qkv attention
        qk_scale : Whether to enable scale factor when calculating the attention.
        attn_drop : The attention dropout
        proj_drop : The projection dropout
        use_final_proj : Whether to use the final projection or not
        norm_layer : The normalization layer
        use_global_vector : Whether to use the global vector or not.
        use_global_self_attn : Whether to do self attention among global vectors
        separate_global_qkv : Whether to different network to calc q_global, k_global, v_global
        global_dim_ratio : The dim (channels) of global vectors is `global_dim_ratio*dim`.
        checkpoint_level : Whether to enable gradient checkpointing.
    """

    def __init__(
        self,
        dim,
        num_heads,
        cuboid_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        strategy=("l", "l", "l"),
        padding_type="ignore",
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_final_proj=True,
        norm_layer="layer_norm",
        use_global_vector=False,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        checkpoint_level=True,
        use_relative_pos=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidSelfAttentionLayer, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.cuboid_size = cuboid_size
        self.shift_size = shift_size
        self.strategy = strategy
        self.padding_type = padding_type
        self.use_final_proj = use_final_proj
        self.use_relative_pos = use_relative_pos
        self.use_global_vector = use_global_vector
        self.use_global_self_attn = use_global_self_attn
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert (
                separate_global_qkv is True
            ), "Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        assert self.padding_type in ["ignore", "zeros", "nearest"]
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if use_relative_pos:
            init_data = paddle.zeros(
                (
                    (2 * cuboid_size[0] - 1)
                    * (2 * cuboid_size[1] - 1)
                    * (2 * cuboid_size[2] - 1),
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

            coords_t = paddle.arange(end=self.cuboid_size[0])
            coords_h = paddle.arange(end=self.cuboid_size[1])
            coords_w = paddle.arange(end=self.cuboid_size[2])
            coords = paddle.stack(x=paddle.meshgrid(coords_t, coords_h, coords_w))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.transpose(perm=[1, 2, 0])
            relative_coords[:, :, 0] += self.cuboid_size[0] - 1
            relative_coords[:, :, 1] += self.cuboid_size[1] - 1
            relative_coords[:, :, 2] += self.cuboid_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.cuboid_size[1] - 1) * (
                2 * self.cuboid_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.cuboid_size[2] - 1
            relative_position_index = relative_coords.sum(axis=-1)
            self.register_buffer(
                name="relative_position_index", tensor=relative_position_index
            )
        self.qkv = paddle.nn.Linear(
            in_features=dim, out_features=dim * 3, bias_attr=qkv_bias
        )
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
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
                self.g2l_global_q_net = paddle.nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim,
                    bias_attr=qkv_bias,
                )
                self.g2l_k_net = paddle.nn.Linear(
                    in_features=dim, out_features=dim, bias_attr=qkv_bias
                )
                self.g2l_v_net = paddle.nn.Linear(
                    in_features=dim,
                    out_features=global_dim_ratio * dim,
                    bias_attr=qkv_bias,
                )
                if self.use_global_self_attn:
                    self.g2g_global_qkv_net = paddle.nn.Linear(
                        in_features=global_dim_ratio * dim,
                        out_features=global_dim_ratio * dim * 3,
                        bias_attr=qkv_bias,
                    )
            else:
                self.global_qkv = paddle.nn.Linear(
                    in_features=dim, out_features=dim * 3, bias_attr=qkv_bias
                )
            self.global_attn_drop = paddle.nn.Dropout(p=attn_drop)
        if use_final_proj:
            self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
            self.proj_drop = paddle.nn.Dropout(p=proj_drop)
            if self.use_global_vector:
                self.global_proj = paddle.nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=global_dim_ratio * dim,
                )
        self.norm = cuboid_utils.get_norm_layer(norm_layer, in_channels=dim)
        if self.use_global_vector:
            self.global_vec_norm = cuboid_utils.get_norm_layer(
                norm_layer, in_channels=global_dim_ratio * dim
            )
        self.checkpoint_level = checkpoint_level
        self.reset_parameters()

    def reset_parameters(self):
        cuboid_utils.apply_initialization(
            self.qkv, linear_mode=self.attn_linear_init_mode
        )
        if self.use_final_proj:
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
                cuboid_utils.apply_initialization(
                    self.g2l_global_q_net, linear_mode=self.attn_linear_init_mode
                )
                cuboid_utils.apply_initialization(
                    self.g2l_k_net, linear_mode=self.attn_linear_init_mode
                )
                cuboid_utils.apply_initialization(
                    self.g2l_v_net, linear_mode=self.attn_linear_init_mode
                )
                if self.use_global_self_attn:
                    cuboid_utils.apply_initialization(
                        self.g2g_global_qkv_net, linear_mode=self.attn_linear_init_mode
                    )
            else:
                cuboid_utils.apply_initialization(
                    self.global_qkv, linear_mode=self.attn_linear_init_mode
                )
            cuboid_utils.apply_initialization(
                self.global_vec_norm, norm_mode=self.norm_init_mode
            )

    def forward(self, x, global_vectors=None):
        x = self.norm(x)

        B, T, H, W, C_in = x.shape
        assert C_in == self.dim
        if self.use_global_vector:
            _, num_global, _ = global_vectors.shape
            global_vectors = self.global_vec_norm(global_vectors)
        cuboid_size, shift_size = update_cuboid_size_shift_size(
            (T, H, W), self.cuboid_size, self.shift_size, self.strategy
        )

        pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
        pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
        pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
        x = cuboid_utils.generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type)

        if any(i > 0 for i in shift_size):
            shifted_x = paddle.roll(
                x=x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                axis=(1, 2, 3),
            )
        else:
            shifted_x = x

        reordered_x = cuboid_reorder(
            shifted_x, cuboid_size=cuboid_size, strategy=self.strategy
        )

        _, num_cuboids, cuboid_volume, _ = reordered_x.shape
        attn_mask = compute_cuboid_self_attention_mask(
            (T, H, W),
            cuboid_size,
            shift_size=shift_size,
            strategy=self.strategy,
            padding_type=self.padding_type,
            device=x.place,
        )
        head_C = C_in // self.num_heads
        qkv = (
            self.qkv(reordered_x)
            .reshape([B, num_cuboids, cuboid_volume, 3, self.num_heads, head_C])
            .transpose(perm=[3, 0, 4, 1, 2, 5])
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        perm_0 = list(range(k.ndim))
        perm_0[-2] = -1
        perm_0[-1] = -2
        attn_score = q @ k.transpose(perm=perm_0)

        if self.use_relative_pos:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:cuboid_volume, :cuboid_volume].reshape(
                    [-1]
                )
            ].reshape([cuboid_volume, cuboid_volume, -1])
            relative_position_bias = relative_position_bias.transpose(
                perm=[2, 0, 1]
            ).unsqueeze(axis=1)
            attn_score = attn_score + relative_position_bias

        if self.use_global_vector:
            global_head_C = self.global_dim_ratio * head_C
            if self.separate_global_qkv:
                l2g_q = (
                    self.l2g_q_net(reordered_x)
                    .reshape([B, num_cuboids, cuboid_volume, self.num_heads, head_C])
                    .transpose(perm=[0, 3, 1, 2, 4])
                )
                l2g_q = l2g_q * self.scale
                l2g_global_kv = (
                    self.l2g_global_kv_net(global_vectors)
                    .reshape([B, 1, num_global, 2, self.num_heads, head_C])
                    .transpose(perm=[3, 0, 4, 1, 2, 5])
                )
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
                g2l_global_q = (
                    self.g2l_global_q_net(global_vectors)
                    .reshape([B, num_global, self.num_heads, head_C])
                    .transpose(perm=[0, 2, 1, 3])
                )
                g2l_global_q = g2l_global_q * self.scale
                g2l_k = (
                    self.g2l_k_net(reordered_x)
                    .reshape([B, num_cuboids, cuboid_volume, self.num_heads, head_C])
                    .transpose(perm=[0, 3, 1, 2, 4])
                )
                g2l_v = (
                    self.g2l_v_net(reordered_x)
                    .reshape(
                        [B, num_cuboids, cuboid_volume, self.num_heads, global_head_C]
                    )
                    .transpose(perm=[0, 3, 1, 2, 4])
                )
                if self.use_global_self_attn:
                    g2g_global_qkv = (
                        self.g2g_global_qkv_net(global_vectors)
                        .reshape([B, 1, num_global, 3, self.num_heads, global_head_C])
                        .transpose(perm=[3, 0, 4, 1, 2, 5])
                    )
                    g2g_global_q, g2g_global_k, g2g_global_v = (
                        g2g_global_qkv[0],
                        g2g_global_qkv[1],
                        g2g_global_qkv[2],
                    )
                    g2g_global_q = g2g_global_q.squeeze(axis=2) * self.scale
            else:
                q_global, k_global, v_global = (
                    self.global_qkv(global_vectors)
                    .reshape([B, 1, num_global, 3, self.num_heads, head_C])
                    .transpose(perm=[3, 0, 4, 1, 2, 5])
                )
                q_global = q_global.squeeze(axis=2) * self.scale
                l2g_q, g2l_k, g2l_v = q, k, v
                g2l_global_q, l2g_global_k, l2g_global_v = (
                    q_global,
                    k_global,
                    v_global,
                )
                if self.use_global_self_attn:
                    g2g_global_q, g2g_global_k, g2g_global_v = (
                        q_global,
                        k_global,
                        v_global,
                    )

            perm_1 = list(range(l2g_global_k.ndim))
            perm_1[-2] = -1
            perm_1[-1] = -2
            l2g_attn_score = l2g_q @ l2g_global_k.transpose(perm=perm_1)
            attn_score_l2l_l2g = paddle.concat(x=(attn_score, l2g_attn_score), axis=-1)

            if attn_mask.ndim == 5:
                attn_mask_l2l_l2g = F.pad(
                    attn_mask, [0, num_global], "constant", 1, data_format="NDHWC"
                )
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.astype("float32")
                attn_mask_l2l_l2g = F.pad(
                    attn_mask, [0, num_global], "constant", 1, data_format="NCL"
                )
                attn_mask_l2l_l2g = attn_mask_l2l_l2g.astype("bool")
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
            attn_score_l2l_l2g = masked_softmax(
                attn_score_l2l_l2g, mask=attn_mask_l2l_l2g
            )
            attn_score_l2l_l2g = self.attn_drop(attn_score_l2l_l2g)
            reordered_x = (
                (attn_score_l2l_l2g @ v_l_g)
                .transpose(perm=[0, 2, 3, 1, 4])
                .reshape([B, num_cuboids, cuboid_volume, self.dim])
            )
            if self.padding_type == "ignore":
                g2l_attn_mask = paddle.ones(shape=(1, T, H, W, 1))
                if pad_t > 0 or pad_h > 0 or pad_w > 0:
                    g2l_attn_mask = F.pad(
                        g2l_attn_mask,
                        [0, 0, 0, pad_w, 0, pad_h, 0, pad_t],
                        data_format="NDHWC",
                    )
                if any(i > 0 for i in shift_size):
                    g2l_attn_mask = paddle.roll(
                        x=g2l_attn_mask,
                        shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                        axis=(1, 2, 3),
                    )
                g2l_attn_mask = g2l_attn_mask.reshape((-1,))
            else:
                g2l_attn_mask = None
            temp = g2l_k.reshape(
                [B, self.num_heads, num_cuboids * cuboid_volume, head_C]
            )
            perm_2 = list(range(temp.ndim))
            perm_2[-2] = -1
            perm_2[-1] = -2
            g2l_attn_score = g2l_global_q @ temp.transpose(perm=perm_2)
            if self.use_global_self_attn:
                temp = g2g_global_k.squeeze(axis=2)
                perm_3 = list(range(temp.ndim))
                perm_3[-2] = -1
                perm_3[-1] = -2
                g2g_attn_score = g2g_global_q @ temp.transpose(perm=perm_3)
                g2all_attn_score = paddle.concat(
                    x=(g2l_attn_score, g2g_attn_score), axis=-1
                )
                if g2l_attn_mask is not None:
                    g2all_attn_mask = F.pad(
                        g2l_attn_mask,
                        [0, num_global],
                        "constant",
                        1,
                        data_format="NDHWC",
                    )
                else:
                    g2all_attn_mask = None
                new_v = paddle.concat(
                    x=(
                        g2l_v.reshape(
                            [
                                B,
                                self.num_heads,
                                num_cuboids * cuboid_volume,
                                global_head_C,
                            ]
                        ),
                        g2g_global_v.reshape(
                            [B, self.num_heads, num_global, global_head_C]
                        ),
                    ),
                    axis=2,
                )
            else:
                g2all_attn_score = g2l_attn_score
                g2all_attn_mask = g2l_attn_mask
                new_v = g2l_v.reshape(
                    [B, self.num_heads, num_cuboids * cuboid_volume, global_head_C]
                )
            g2all_attn_score = masked_softmax(g2all_attn_score, mask=g2all_attn_mask)
            g2all_attn_score = self.global_attn_drop(g2all_attn_score)
            new_global_vector = (
                (g2all_attn_score @ new_v)
                .transpose(perm=[0, 2, 1, 3])
                .reshape([B, num_global, self.global_dim_ratio * self.dim])
            )
        else:
            attn_score = masked_softmax(attn_score, mask=attn_mask)
            attn_score = self.attn_drop(attn_score)
            reordered_x = (
                (attn_score @ v)
                .transpose(perm=[0, 2, 3, 1, 4])
                .reshape([B, num_cuboids, cuboid_volume, self.dim])
            )

        if self.use_final_proj:
            reordered_x = paddle.cast(reordered_x, dtype="float32")
            reordered_x = self.proj_drop(self.proj(reordered_x))
            if self.use_global_vector:
                new_global_vector = self.proj_drop(self.global_proj(new_global_vector))
        shifted_x = cuboid_reorder_reverse(
            reordered_x,
            cuboid_size=cuboid_size,
            strategy=self.strategy,
            orig_data_shape=(T + pad_t, H + pad_h, W + pad_w),
        )
        if any(i > 0 for i in shift_size):
            x = paddle.roll(
                x=shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                axis=(1, 2, 3),
            )
        else:
            x = shifted_x
        x = cuboid_utils.generalize_unpadding(
            x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type
        )
        if self.use_global_vector:
            return x, new_global_vector
        else:
            return x


class StackCuboidSelfAttentionBlock(paddle.nn.Layer):
    """

    - "use_inter_ffn" is True
        x --> attn1 -----+-------> ffn1 ---+---> attn2 --> ... --> ffn_k --> out
           |             ^   |             ^
           |             |   |             |
           |-------------|   |-------------|
    - "use_inter_ffn" is False
        x --> attn1 -----+------> attn2 --> ... attnk --+----> ffnk ---+---> out
           |             ^   |            ^             ^  |           ^
           |             |   |            |             |  |           |
           |-------------|   |------------|   ----------|  |-----------|
    If we have enabled global memory vectors, each attention will be a

    """

    def __init__(
        self,
        dim,
        num_heads,
        block_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        block_shift_size=[(0, 0, 0), (2, 2, 2)],
        block_strategy=[("d", "d", "d"), ("l", "l", "l")],
        padding_type="ignore",
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        use_inter_ffn=False,
        use_global_vector=False,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        checkpoint_level=True,
        use_relative_pos=True,
        use_final_proj=True,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(StackCuboidSelfAttentionBlock, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode
        if (
            len(block_cuboid_size[0]) <= 0
            or len(block_shift_size) <= 0
            or len(block_strategy) <= 0
        ):
            raise ValueError(
                "Format of the block cuboid size is not correct. block_cuboid_size={block_cuboid_size}"
            )
        if len(block_cuboid_size) != len(block_shift_size) and len(
            block_cuboid_size
        ) != len(block_strategy):
            raise ValueError(
                "The lengths of block_cuboid_size, block_shift_size, and block_strategy must be equal."
            )

        self.num_attn = len(block_cuboid_size)
        self.checkpoint_level = checkpoint_level
        self.use_inter_ffn = use_inter_ffn
        self.use_global_vector = use_global_vector
        self.use_global_vector_ffn = use_global_vector_ffn
        self.use_global_self_attn = use_global_self_attn
        self.global_dim_ratio = global_dim_ratio
        if self.use_inter_ffn:
            self.ffn_l = paddle.nn.LayerList(
                sublayers=[
                    PositionwiseFFN(
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
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = paddle.nn.LayerList(
                    sublayers=[
                        PositionwiseFFN(
                            units=global_dim_ratio * dim,
                            hidden_size=global_dim_ratio * 4 * dim,
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
                    PositionwiseFFN(
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
            if self.use_global_vector_ffn and self.use_global_vector:
                self.global_ffn_l = paddle.nn.LayerList(
                    sublayers=[
                        PositionwiseFFN(
                            units=global_dim_ratio * dim,
                            hidden_size=global_dim_ratio * 4 * dim,
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
                CuboidSelfAttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    cuboid_size=ele_cuboid_size,
                    shift_size=ele_shift_size,
                    strategy=ele_strategy,
                    padding_type=padding_type,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    use_global_vector=use_global_vector,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    checkpoint_level=checkpoint_level,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=use_final_proj,
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for ele_cuboid_size, ele_shift_size, ele_strategy in zip(
                    block_cuboid_size, block_shift_size, block_strategy
                )
            ]
        )

    def reset_parameters(self):
        for m in self.ffn_l:
            m.reset_parameters()
        if self.use_global_vector_ffn and self.use_global_vector:
            for m in self.global_ffn_l:
                m.reset_parameters()
        for m in self.attn_l:
            m.reset_parameters()

    def forward(self, x, global_vectors=None):
        if self.use_inter_ffn:
            if self.use_global_vector:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x_out, global_vectors_out = fleet.utils.recompute(
                            attn, x, global_vectors
                        )
                    else:
                        x_out, global_vectors_out = attn(x, global_vectors)
                    x = x + x_out
                    global_vectors = global_vectors + global_vectors_out
                    if self.checkpoint_level >= 1 and self.training:
                        x = fleet.utils.recompute(ffn, x)
                        if self.use_global_vector_ffn:
                            global_vectors = fleet.utils.recompute(
                                self.global_ffn_l[idx], global_vectors
                            )
                    else:
                        x = ffn(x)
                        if self.use_global_vector_ffn:
                            global_vectors = self.global_ffn_l[idx](global_vectors)
                return x, global_vectors
            else:
                for idx, (attn, ffn) in enumerate(zip(self.attn_l, self.ffn_l)):
                    if self.checkpoint_level >= 2 and self.training:
                        x = x + fleet.utils.recompute(attn, x)
                    else:
                        x = x + attn(x)
                    if self.checkpoint_level >= 1 and self.training:
                        x = fleet.utils.recompute(ffn, x)
                    else:
                        x = ffn(x)
                return x
        elif self.use_global_vector:
            for idx, attn in enumerate(self.attn_l):
                if self.checkpoint_level >= 2 and self.training:
                    x_out, global_vectors_out = fleet.utils.recompute(
                        attn, x, global_vectors
                    )
                else:
                    x_out, global_vectors_out = attn(x, global_vectors)
                x = x + x_out
                global_vectors = global_vectors + global_vectors_out
            if self.checkpoint_level >= 1 and self.training:
                x = fleet.utils.recompute(self.ffn_l[0], x)
                if self.use_global_vector_ffn:
                    global_vectors = fleet.utils.recompute(
                        self.global_ffn_l[0], global_vectors
                    )
            else:
                x = self.ffn_l[0](x)
                if self.use_global_vector_ffn:
                    global_vectors = self.global_ffn_l[0](global_vectors)
            return x, global_vectors
        else:
            for idx, attn in enumerate(self.attn_l):
                if self.checkpoint_level >= 2 and self.training:
                    out = fleet.utils.recompute(attn, x)
                else:
                    out = attn(x)
                x = x + out
            if self.checkpoint_level >= 1 and self.training:
                x = fleet.utils.recompute(self.ffn_l[0], x)
            else:
                x = self.ffn_l[0](x)
            return x


class CuboidTransformerEncoder(paddle.nn.Layer):
    """Encoder of the CuboidTransformer

    x --> attn_block --> patch_merge --> attn_block --> patch_merge --> ... --> out

    Args:
        input_shape : The shape of the input. Contains T, H, W, C
        initial_data_thw : The shape of the first layer
        base_units : The number of units
        scale_alpha : We scale up the channels based on the formula:
            - round_to(base_units * max(downsample_scale) ** units_alpha, 4)
        depth : The number of layers for each block
        downsample : The downsample ratio
        downsample_type : Type of the downsampling layer
        block_attn_patterns : Attention pattern for the cuboid attention for each block.
        block_cuboid_size : A list of cuboid size parameters
        block_strategy : A list of cuboid strategies
        block_shift_size : A list of shift sizes
        num_global : The number of global vectors
        num_heads : The number of heads.
        attn_drop
        proj_drop
        ffn_drop
        gated_ffn : Whether to enable gated ffn or not
        norm_layer : The normalization layer
        use_inter_ffn : Whether to use intermediate FFN
        padding_type
    """

    def __init__(
        self,
        input_shape,
        base_units=128,
        block_units=None,
        scale_alpha=1.0,
        depth=[4, 4, 4],
        downsample=2,
        downsample_type="patch_merge",
        block_attn_patterns=None,
        block_cuboid_size=[(4, 4, 4), (4, 4, 4)],
        block_strategy=[("l", "l", "l"), ("d", "d", "d")],
        block_shift_size=[(0, 0, 0), (0, 0, 0)],
        num_heads=4,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        activation="leaky",
        ffn_activation="leaky",
        gated_ffn=False,
        norm_layer="layer_norm",
        use_inter_ffn=True,
        padding_type="ignore",
        checkpoint_level=True,
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        use_global_vector=False,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        global_dim_ratio=1,
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        conv_init_mode="0",
        down_linear_init_mode="0",
        norm_init_mode="0",
    ):
        super(CuboidTransformerEncoder, self).__init__()
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_linear_init_mode = down_linear_init_mode
        self.norm_init_mode = norm_init_mode
        self.input_shape = input_shape
        self.depth = depth
        self.num_blocks = len(depth)
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        if not isinstance(downsample, (tuple, list)):
            downsample = 1, downsample, downsample
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.num_heads = num_heads
        self.use_global_vector = use_global_vector
        self.checkpoint_level = checkpoint_level
        if block_units is None:
            block_units = [
                cuboid_utils.round_to(
                    base_units * int((max(downsample) ** scale_alpha) ** i), 4
                )
                for i in range(self.num_blocks)
            ]
        else:
            assert len(block_units) == self.num_blocks and block_units[0] == base_units
        self.block_units = block_units
        if self.num_blocks > 1:
            if downsample_type == "patch_merge":
                self.down_layers = paddle.nn.LayerList(
                    sublayers=[
                        PatchMerging3D(
                            dim=self.block_units[i],
                            downsample=downsample,
                            padding_type=padding_type,
                            out_dim=self.block_units[i + 1],
                            linear_init_mode=down_linear_init_mode,
                            norm_init_mode=norm_init_mode,
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            else:
                raise NotImplementedError(f"{downsample_type} is invalid.")
            if self.use_global_vector:
                self.down_layer_global_proj = paddle.nn.LayerList(
                    sublayers=[
                        paddle.nn.Linear(
                            in_features=global_dim_ratio * self.block_units[i],
                            out_features=global_dim_ratio * self.block_units[i + 1],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
        if block_attn_patterns is not None:
            mem_shapes = self.get_mem_shapes()
            if isinstance(block_attn_patterns, (tuple, list)):
                assert len(block_attn_patterns) == self.num_blocks
            else:
                block_attn_patterns = [
                    block_attn_patterns for _ in range(self.num_blocks)
                ]
            block_cuboid_size = []
            block_strategy = []
            block_shift_size = []
            for idx, key in enumerate(block_attn_patterns):
                func = cuboid_utils.CuboidSelfAttentionPatterns.get(key)
                cuboid_size, strategy, shift_size = func(mem_shapes[idx])
                block_cuboid_size.append(cuboid_size)
                block_strategy.append(strategy)
                block_shift_size.append(shift_size)
        else:
            if not isinstance(block_cuboid_size[0][0], (list, tuple)):
                block_cuboid_size = [block_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert (
                    len(block_cuboid_size) == self.num_blocks
                ), f"Incorrect input format! Received block_cuboid_size={block_cuboid_size}"
            if not isinstance(block_strategy[0][0], (list, tuple)):
                block_strategy = [block_strategy for _ in range(self.num_blocks)]
            else:
                assert (
                    len(block_strategy) == self.num_blocks
                ), f"Incorrect input format! Received block_strategy={block_strategy}"
            if not isinstance(block_shift_size[0][0], (list, tuple)):
                block_shift_size = [block_shift_size for _ in range(self.num_blocks)]
            else:
                assert (
                    len(block_shift_size) == self.num_blocks
                ), f"Incorrect input format! Received block_shift_size={block_shift_size}"
        self.block_cuboid_size = block_cuboid_size
        self.block_strategy = block_strategy
        self.block_shift_size = block_shift_size
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Sequential(
                    *[
                        StackCuboidSelfAttentionBlock(
                            dim=self.block_units[i],
                            num_heads=num_heads,
                            block_cuboid_size=block_cuboid_size[i],
                            block_strategy=block_strategy[i],
                            block_shift_size=block_shift_size[i],
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            ffn_drop=ffn_drop,
                            activation=ffn_activation,
                            gated_ffn=gated_ffn,
                            norm_layer=norm_layer,
                            use_inter_ffn=use_inter_ffn,
                            padding_type=padding_type,
                            use_global_vector=use_global_vector,
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
                        for _ in range(depth[i])
                    ]
                )
                for i in range(self.num_blocks)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_blocks > 1:
            for m in self.down_layers:
                m.reset_parameters()
            if self.use_global_vector:
                cuboid_utils.apply_initialization(
                    self.down_layer_global_proj, linear_mode=self.down_linear_init_mode
                )
        for ms in self.blocks:
            for m in ms:
                m.reset_parameters()

    def get_mem_shapes(self):
        """Get the shape of the output memory based on the input shape. This can be used for constructing the decoder.

        Returns:
            mem_shapes : A list of shapes of the output memory
        """

        if self.num_blocks == 1:
            return [self.input_shape]
        else:
            mem_shapes = [self.input_shape]
            curr_shape = self.input_shape
            for down_layer in self.down_layers:
                curr_shape = down_layer.get_out_shape(curr_shape)
                mem_shapes.append(curr_shape)
            return mem_shapes

    def forward(self, x, global_vectors=None):
        """
        Args:
            x : Shape (B, T, H, W, C)

        Returns:
            out : A list of tensors from the bottom layer to the top layer of the encoder. For example, it can have shape
                - (B, T, H, W, C1)
                - (B, T, H // 2, W // 2, 2 * C1)
                - (B, T, H // 4, W // 4, 4 * C1)
                ...
            global_mem_out : Optional
        """

        B, T, H, W, C_in = x.shape
        assert (T, H, W, C_in) == self.input_shape

        if self.use_global_vector:
            out = []
            global_mem_out = []
            for i in range(self.num_blocks):
                for l in self.blocks[i]:
                    x, global_vectors = l(x, global_vectors)
                out.append(x)
                global_mem_out.append(global_vectors)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
                    global_vectors = self.down_layer_global_proj[i](global_vectors)
            return out, global_mem_out
        else:
            out = []
            for i in range(self.num_blocks):
                x = self.blocks[i](x)
                out.append(x)
                if self.num_blocks > 1 and i < self.num_blocks - 1:
                    x = self.down_layers[i](x)
            return out

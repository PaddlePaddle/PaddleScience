# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code below is heavily based on [transformer-physx](https://github.com/zabaras/transformer-physx)
"""

from typing import Optional
from typing import Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.initializer import Constant
from paddle.nn.initializer import Normal

from ppsci.arch import base

zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class MaskedAttention(nn.Layer):
    """Masked self-attention module.

    Args:
        embed_dim (int): The expected feature size in the input and output.
        num_ctx (int): Contex length of block.
        num_heads (int): The number of heads in multi-head attention.
        attn_drop (float, optional): The dropout probability used on attention
            weights to drop some attention targets. Defaults to 0.
        proj_drop (float, optional): The dropout probability used on output. Defaults to 0.
        scale (bool, optional): Whether to scale attention weights. Defaults to False.
    """

    def __init__(
        self,
        embed_dim: int,
        num_ctx: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale: bool = False,
    ):
        super().__init__()
        self.register_buffer(
            "bias",
            paddle.tril(paddle.ones((num_ctx, num_ctx), dtype="int32")).reshape(
                [1, 1, num_ctx, num_ctx]
            ),
        )

        self.register_buffer("masked_bias", paddle.to_tensor(-1e4))
        self.num_heads = num_heads
        self.split_size = embed_dim
        self.scale = scale

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        attn = paddle.matmul(query, key)
        if self.scale:
            attn = attn / (float(value.shape[-1]) ** 0.5)

        nd, ns = attn.shape[-2], attn.shape[-1]
        mask = self.bias[:, :, ns - nd : ns, :ns]
        attn = paddle.where(mask > 0, attn, self.masked_bias.cast(attn.dtype))

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        if head_mask is not None:
            attn = attn * head_mask

        outputs = [paddle.matmul(attn, value)]
        if output_attentions:
            outputs.append(attn)
        return outputs

    def merge_heads(self, x):
        x = x.transpose([0, 2, 1, 3])
        new_x_shape = x.shape[:-2] + [
            x.shape[-2] * x.shape[-1],
        ]
        return x.reshape(new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.shape[:-1] + [self.num_heads, x.shape[-1] // self.num_heads]
        x = x.reshape(new_x_shape)
        if k:
            return x.transpose([0, 2, 3, 1])
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        x = self.qkv_proj(x)
        query, key, value = x.split(x.shape[2] // self.split_size, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        # Concat previous key and value tensors
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose([0, 1, 3, 2]), layer_past[1]
            key = paddle.concat((past_key, key), axis=-1)
            value = paddle.concat((past_value, value), axis=-2)

        attn_outputs = self._attn(
            query, key, value, attention_mask, head_mask, output_attentions
        )
        output = attn_outputs[0]
        output = self.merge_heads(output)
        output = self.out_proj(output)
        output = self.proj_drop(output)

        outputs = [output] + attn_outputs[1:]
        return outputs


class MLP(nn.Layer):
    """Multi layer perceptron module used in Transformer.

    Args:
        in_features (int): Number of the input features.
        hidden_features (Optional[int]): Number of the hidden size. Defaults to None.
        out_features (Optional[int]): Number of the output features. Defaults to None.
        drop (float, optional): Probability of dropout the units. Defaults to 0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Layer):
    """Transformer decoder block consisting of layer norm,
        masked self-attention, layer norm and fully connected layer.

    Args:
        num_ctx (int): Contex length of block
        embed_size (int): The number of embedding size.
        num_heads (int): The number of heads in multi-head attention.
        attn_pdrop (float): The dropout probability used on attention
            weights to drop some attention targets.
        resid_pdrop (float): The dropout probability used on output.
        scale (bool, optional): Scaled self-attention calculation. Defaults to False.
    """

    def __init__(
        self,
        num_ctx: int,
        embed_size: int,
        num_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        scale: bool = False,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = MaskedAttention(
            embed_size, num_ctx, num_heads, attn_pdrop, resid_pdrop, scale
        )
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, 4 * embed_size, resid_pdrop)

    def forward(
        self,
        x,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        x = x + output_attn[0]
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [x] + output_attn[1:]
        return outputs


class PhysformerGPT2(base.Arch):
    """Transformer decoder model for modeling physics.

    Args:
        input_keys (Tuple[str, ...]): Input keys, such as ("embeds",).
        output_keys (Tuple[str, ...]): Output keys, such as ("pred_embeds",).
        num_layers (int): Number of transformer layers.
        num_ctx (int): Contex length of block.
        embed_size (int): The number of embedding size.
        num_heads (int): The number of heads in multi-head attention.
        embd_pdrop (float, optional): The dropout probability used on embedding features. Defaults to 0.0.
        attn_pdrop (float, optional): The dropout probability used on attention weights. Defaults to 0.0.
        resid_pdrop (float, optional): The dropout probability used on block outputs. Defaults to 0.0.
        initializer_range (float, optional): Initializer range of linear layer. Defaults to 0.05.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.PhysformerGPT2(("embeds", ), ("pred_embeds", ), 6, 16, 128, 4)
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        num_ctx: int,
        embed_size: int,
        num_heads: int,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        initializer_range: float = 0.05,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.num_layers = num_layers
        self.num_ctx = num_ctx
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.initializer_range = initializer_range

        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.LayerList(
            [
                Block(
                    num_ctx, embed_size, num_heads, attn_pdrop, resid_pdrop, scale=True
                )
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, embed_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_ = Normal(mean=0.0, std=self.initializer_range)
            normal_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)

    def get_position_embed(self, x):
        B, N, _ = x.shape
        position_ids = paddle.arange(0, N, dtype=paddle.get_default_dtype()).reshape(
            [1, N, 1]
        )
        position_ids = position_ids.repeat_interleave(B, axis=0)

        position_embeds = paddle.zeros_like(x)
        i = paddle.arange(0, self.embed_size // 2).unsqueeze(0).unsqueeze(0)
        position_embeds[:, :, ::2] = paddle.sin(
            position_ids / 10000 ** (2 * i / self.embed_size)
        )
        position_embeds[:, :, 1::2] = paddle.cos(
            position_ids / 10000 ** (2 * i / self.embed_size)
        )
        return position_embeds

    def _generate_time_series(self, x, max_length):
        cur_len = x.shape[1]
        if cur_len >= max_length:
            raise ValueError(
                f"max_length({max_length}) should be larger than "
                f"the length of input context({cur_len})"
            )

        while cur_len < max_length:
            model_inputs = x[:, -1:]
            outputs = self.forward_tensor(model_inputs)
            next_output = outputs[0][:, -1:]
            x = paddle.concat([x, next_output], axis=1)
            cur_len = cur_len + 1
        return x

    @paddle.no_grad()
    def generate(self, x, max_length=256):
        if max_length <= 0:
            raise ValueError(
                "max_length({max_length}) should be a strictly positive integer."
            )
        outputs = self._generate_time_series(x, max_length)
        return outputs

    def forward_tensor(self, x):
        position_embeds = self.get_position_embed(x)
        # Combine input embedding, position embeding
        hidden_states = x + position_embeds
        hidden_states = self.drop(hidden_states)

        # Loop through transformer self-attention layers
        for block in self.blocks:
            block_outputs = block(hidden_states)
            hidden_states = block_outputs[0]
        outputs = self.linear(self.ln(hidden_states))
        return (outputs,)

    def forward_eval(self, x):
        input_embeds = x[:, :1]
        outputs = self.generate(input_embeds)
        return (outputs[:, 1:],)

    def split_to_dict(self, data_tensors, keys):
        return {key: data_tensors[i] for i, key in enumerate(keys)}

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)
        x = self.concat_to_tensor(x, self.input_keys, axis=-1)
        if self.training:
            y = self.forward_tensor(x)
        else:
            y = self.forward_eval(x)
        y = self.split_to_dict(y, self.output_keys)
        if self._output_transform is not None:
            y = self._output_transform(y)
        return y

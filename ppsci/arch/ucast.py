# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import math
from typing import Sequence

import paddle
from paddle import nn
from paddle.nn import functional as F

from ppsci.arch import base


class _UCastConv2D(nn.Layer):
    """U-Cast convolution with optional nearest-up/average-down sampling.

    Longitude is stored on the height axis in the WeatherBench-2 data used by
    U-Cast. Convolutions therefore use circular padding on height and zero
    padding on width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        up: bool = False,
        down: bool = False,
        init_zero: bool = False,
    ):
        super().__init__()
        if up and down:
            raise ValueError("up and down cannot both be True")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.padding = kernel_size // 2 if kernel_size else 0

        if kernel_size:
            fan_in = in_channels * kernel_size**2
            # U-Cast uses the EDM ``kaiming_uniform`` initializer scaled by
            # sqrt(1 / 3), which is Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
            bound = 0.0 if init_zero else 1.0 / math.sqrt(fan_in)
            self.weight = self.create_parameter(
                [out_channels, in_channels, kernel_size, kernel_size],
                default_initializer=nn.initializer.Uniform(-bound, bound),
            )
            self.bias = self.create_parameter(
                [out_channels],
                default_initializer=(
                    nn.initializer.Constant(0.0)
                    if init_zero
                    else nn.initializer.Uniform(-bound, bound)
                ),
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.up:
            # Paddle's AMP blacklist promotes interpolation, while the source
            # preserves its input dtype (including FP16 skip connections).
            with paddle.amp.auto_cast(enable=False):
                x = F.interpolate(x, scale_factor=2, mode="nearest", data_format="NCHW")
        if self.down:
            x = F.avg_pool2d(x, kernel_size=2, data_format="NCHW")

        if self.weight is not None:
            if self.padding:
                # Circular padding on longitude (H), zero padding on latitude (W).
                x = F.pad(
                    x,
                    [0, 0, self.padding, self.padding],
                    mode="circular",
                    data_format="NCHW",
                )
                x = F.pad(
                    x,
                    [self.padding, self.padding, 0, 0],
                    mode="constant",
                    value=0.0,
                    data_format="NCHW",
                )
            # The source adds bias after the convolution's output is rounded.
            # Fusing bias into cuDNN changes FP16 values and downstream gradients.
            bias = self.bias.astype(x.dtype)
            x = F.conv2d(x, self.weight, data_format="NCHW")
            output_dtype = x.dtype
            x = (x.astype(bias.dtype) + bias.reshape([1, -1, 1, 1])).astype(
                output_dtype
            )
        return x


class _UCastGroupNorm(nn.Layer):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        min_channels_per_group: int = 4,
    ):
        super().__init__()
        num_groups = 32
        while (
            num_channels % num_groups != 0
            or num_channels // num_groups < min_channels_per_group
        ):
            if num_groups == 1:
                break
            num_groups //= 2
        self.num_groups = num_groups
        self.eps = eps
        self.weight = self.create_parameter(
            [num_channels], default_initializer=nn.initializer.Constant(1.0)
        )
        self.bias = self.create_parameter(
            [num_channels], default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return F.group_norm(
            x,
            num_groups=self.num_groups,
            epsilon=self.eps,
            # The source casts affine parameters before AMP promotes group_norm.
            weight=self.weight.astype(x.dtype),
            bias=self.bias.astype(x.dtype),
            data_format="NCHW",
        )


class _UCastAttentionOp(paddle.autograd.PyLayer):
    """Preserve the original attention's explicit FP32 backward reductions.

    The source saves attention weights after rounding to the query dtype.
    Differentiating through a float32 softmax followed by a cast instead saves
    the unrounded probabilities and changes small FP16 attention gradients.
    """

    @staticmethod
    def forward(ctx, q, k):
        weights = paddle.einsum(
            "ncq,nck->nqk",
            q.astype("float32"),
            (k / math.sqrt(k.shape[1])).astype("float32"),
        )
        weights = F.softmax(weights, axis=2).astype(q.dtype)
        ctx.save_for_backward(q, k, weights)
        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        q, k, weights = ctx.saved_tensor()
        with paddle.amp.auto_cast(enable=False):
            weights = weights.astype("float32")
            grad_weights = grad_weights.astype("float32")
            grad_logits = weights * (
                grad_weights - (grad_weights * weights).sum(axis=2, keepdim=True)
            )
            grad_q = paddle.einsum(
                "nck,nqk->ncq", k.astype("float32"), grad_logits
            ).astype(q.dtype) / math.sqrt(k.shape[1])
            grad_k = paddle.einsum(
                "ncq,nqk->nck", q.astype("float32"), grad_logits
            ).astype(k.dtype) / math.sqrt(k.shape[1])
        return grad_q, grad_k


class _UCastUNetBlock(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )

        self.norm0 = _UCastGroupNorm(in_channels, eps)
        self.conv0 = _UCastConv2D(in_channels, out_channels, 3, up=up, down=down)
        self.norm1 = _UCastGroupNorm(out_channels, eps)
        self.conv1 = _UCastConv2D(out_channels, out_channels, 3, init_zero=True)
        self.dropout = nn.Dropout(dropout)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel_size = 1 if out_channels != in_channels else 0
            self.skip = _UCastConv2D(
                in_channels,
                out_channels,
                kernel_size,
                up=up,
                down=down,
            )

        if self.num_heads:
            self.norm2 = _UCastGroupNorm(out_channels, eps)
            self.qkv = _UCastConv2D(out_channels, out_channels * 3, 1)
            self.proj = _UCastConv2D(out_channels, out_channels, 1, init_zero=True)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x
        x = self.conv0(F.silu(self.norm0(x)))
        x = self.conv1(self.dropout(F.silu(self.norm1(x))))
        # Match the source's in-place additions, which retain the left dtype.
        x = (x + (self.skip(residual) if self.skip is not None else residual)).astype(
            x.dtype
        )

        if self.num_heads:
            batch_size, channels, height, width = x.shape
            channels_per_head = channels // self.num_heads
            qkv = self.qkv(self.norm2(x)).reshape(
                [batch_size * self.num_heads, channels_per_head, 3, -1]
            )
            q, k, v = paddle.unbind(qkv, axis=2)
            attention = _UCastAttentionOp.apply(q, k)
            attended = paddle.einsum("nqk,nck->ncq", attention, v)
            attended = attended.reshape(
                [
                    batch_size,
                    self.num_heads,
                    channels_per_head,
                    height,
                    width,
                ]
            ).reshape([batch_size, channels, height, width])
            projected = self.proj(attended)
            x = (projected + x).astype(projected.dtype)
        return x


class UCast(base.Arch):
    """U-Cast probabilistic weather forecasting network.

    This class implements the ADM/Dhariwal U-Net used by U-Cast. It performs a
    single residual-forecast step; normalization, residual post-processing,
    ensemble sampling and autoregressive rollout are handled by the U-Cast
    example pipeline.

    Args:
        input_keys: Keys concatenated along the channel axis. The standard
            U-Cast pipeline uses ``("input", "dynamical_condition",
            "static_condition")``.
        output_keys: A single output key for the normalized residual.
        in_channels: Total channels after concatenating all inputs.
        out_channels: Number of predicted weather-variable channels.
        model_channels: Base U-Net channel width.
        channel_mult: Channel multiplier at each resolution.
        num_blocks: Residual blocks per encoder resolution.
        attn_levels: Resolution levels that use self-attention.
        channels_per_head: Attention channels per head.
        dropout: MC-Dropout probability.
        image_resolution: Longitude resolution used to name hierarchy levels.
        training_ensemble_size: Number of MC-Dropout members generated together
            during training.
        eval_ensemble_size: Number of MC-Dropout members generated together
            during evaluation.
        use_recompute: Recompute training blocks during backward to reduce memory.
    """

    def __init__(
        self,
        input_keys: tuple[str, ...],
        output_keys: tuple[str, ...],
        in_channels: int,
        out_channels: int,
        model_channels: int = 320,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        num_blocks: int = 4,
        attn_levels: Sequence[int] = (2, 3),
        channels_per_head: int = 64,
        dropout: float = 0.1,
        image_resolution: int = 240,
        training_ensemble_size: int = 1,
        eval_ensemble_size: int = 1,
        use_recompute: bool = False,
    ):
        super().__init__()
        if len(output_keys) != 1:
            raise ValueError("UCast expects exactly one output key")
        if training_ensemble_size < 1 or eval_ensemble_size < 1:
            raise ValueError("UCast ensemble sizes must be positive")
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.training_ensemble_size = training_ensemble_size
        self.eval_ensemble_size = eval_ensemble_size
        self.use_recompute = use_recompute

        block_kwargs = {
            "channels_per_head": channels_per_head,
            "dropout": dropout,
        }
        channel_mult = tuple(channel_mult)
        attn_levels = tuple(attn_levels)

        self.enc = nn.LayerDict()
        current_channels = in_channels
        for level, mult in enumerate(channel_mult):
            resolution = image_resolution >> level
            level_channels = model_channels * mult
            if level == 0:
                current_channels = level_channels
                self.enc[f"{resolution}x{resolution}_conv"] = _UCastConv2D(
                    in_channels, current_channels, 3
                )
            else:
                self.enc[f"{resolution}x{resolution}_down"] = _UCastUNetBlock(
                    current_channels,
                    current_channels,
                    down=True,
                    **block_kwargs,
                )
            for block_idx in range(num_blocks):
                input_channels = current_channels
                current_channels = level_channels
                self.enc[
                    f"{resolution}x{resolution}_block{block_idx}"
                ] = _UCastUNetBlock(
                    input_channels,
                    current_channels,
                    attention=level in attn_levels,
                    **block_kwargs,
                )

        skip_channels = [layer.out_channels for layer in self.enc.values()]
        self.dec = nn.LayerDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            resolution = image_resolution >> level
            level_channels = model_channels * mult
            if level == len(channel_mult) - 1:
                self.dec[f"{resolution}x{resolution}_in0"] = _UCastUNetBlock(
                    current_channels,
                    current_channels,
                    attention=True,
                    **block_kwargs,
                )
                self.dec[f"{resolution}x{resolution}_in1"] = _UCastUNetBlock(
                    current_channels, current_channels, **block_kwargs
                )
            else:
                self.dec[f"{resolution}x{resolution}_up"] = _UCastUNetBlock(
                    current_channels,
                    current_channels,
                    up=True,
                    **block_kwargs,
                )

            for block_idx in range(num_blocks + 1):
                input_channels = current_channels + skip_channels.pop()
                current_channels = level_channels
                self.dec[
                    f"{resolution}x{resolution}_block{block_idx}"
                ] = _UCastUNetBlock(
                    input_channels,
                    current_channels,
                    attention=level in attn_levels,
                    **block_kwargs,
                )

        self.out_norm = _UCastGroupNorm(current_channels)
        self.out_conv = _UCastConv2D(current_channels, out_channels, 3, init_zero=True)

    def _run_block(self, block, x):
        if self.training and self.use_recompute and not x.stop_gradient:
            from paddle.distributed.fleet.utils import recompute

            # Non-reentrant saved-tensor hooks conflict with the custom
            # attention backward in Paddle 3.3; reentrant recomputation keeps
            # its saved queries/keys/probabilities and AMP context consistent.
            return recompute(block, x, use_reentrant=True, preserve_rng_state=True)
        return block(x)

    def forward_tensor(self, inputs: paddle.Tensor) -> paddle.Tensor:
        skips = []
        x = inputs
        for block in self.enc.values():
            x = self._run_block(block, x)
            skips.append(x)

        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                skip = skips.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    with paddle.amp.auto_cast(enable=False):
                        x = F.interpolate(
                            x,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            data_format="NCHW",
                        )
                x = paddle.concat([x, skip], axis=1)
            x = self._run_block(block, x)
        return self.out_conv(F.silu(self.out_norm(x)))

    def forward(self, x: dict[str, paddle.Tensor]) -> dict[str, paddle.Tensor]:
        if self._input_transform is not None:
            x = self._input_transform(x)
        inputs = self.concat_to_tensor(x, self.input_keys, axis=1)
        ensemble_size = (
            self.training_ensemble_size if self.training else self.eval_ensemble_size
        )
        if ensemble_size > 1:
            batch_size = inputs.shape[0]
            inputs = (
                inputs.unsqueeze(0)
                .expand([ensemble_size, -1, -1, -1, -1])
                .reshape([-1, *inputs.shape[1:]])
            )
            enable_eval_dropout = not self.training
            if enable_eval_dropout:
                for layer in self.sublayers():
                    if isinstance(layer, nn.Dropout):
                        layer.train()
            try:
                prediction = self.forward_tensor(inputs)
            finally:
                if enable_eval_dropout:
                    for layer in self.sublayers():
                        if isinstance(layer, nn.Dropout):
                            layer.eval()
            prediction = prediction.reshape(
                [ensemble_size, batch_size, *prediction.shape[1:]]
            )
        else:
            prediction = self.forward_tensor(inputs)
        output = {self.output_keys[0]: prediction}
        if self._output_transform is not None:
            output = self._output_transform(x, output)
        return output

    def enable_mc_dropout(self) -> None:
        """Enable only dropout layers while keeping the rest of the model in eval mode."""
        self.eval()
        for layer in self.sublayers():
            if isinstance(layer, nn.Dropout):
                layer.train()

    def prepare_for_inference_amp(self, dtype: str = "float16") -> None:
        """Cast convolution weights for fast inference while keeping norms in FP32."""
        paddle.amp.decorate(
            models=self,
            level="O2",
            dtype=dtype,
            excluded_layers=_UCastGroupNorm,
        )

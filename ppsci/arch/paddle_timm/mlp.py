# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24).

Linear is native paddle.nn.Linear ([in, out] weight layout; torch Linear
weights must be transposed when transferred).
"""

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import paddle

from .helpers import to_2tuple


class Mlp(paddle.nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[paddle.nn.Module] = paddle.nn.GELU,
        norm_layer: Optional[Type[paddle.nn.Module]] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = (
            partial(paddle.nn.Conv2d, kernel_size=1) if use_conv else paddle.nn.Linear
        )
        self.fc1 = linear_layer(
            in_features, hidden_features, bias_attr=None if bias[0] else False
        )
        self.act = act_layer()
        self.drop1 = paddle.nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features)
            if norm_layer is not None
            else paddle.nn.Identity()
        )
        self.fc2 = linear_layer(
            hidden_features, out_features, bias_attr=None if bias[1] else False
        )
        self.drop2 = paddle.nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

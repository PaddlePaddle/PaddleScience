# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal set of timm components needed by the ClimaX backbone.

# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24), converted to paddle and trimmed to the code paths ClimaX
exercises (manual attention branch, no qk-norm / layer scale).
"""

from ppsci.arch.paddle_timm.attention import Attention
from ppsci.arch.paddle_timm.block import Block
from ppsci.arch.paddle_timm.drop import DropPath
from ppsci.arch.paddle_timm.drop import drop_path
from ppsci.arch.paddle_timm.format import Format
from ppsci.arch.paddle_timm.helpers import to_1tuple
from ppsci.arch.paddle_timm.helpers import to_2tuple
from ppsci.arch.paddle_timm.helpers import to_ntuple
from ppsci.arch.paddle_timm.mlp import Mlp
from ppsci.arch.paddle_timm.patch_embed import PatchEmbed
from ppsci.arch.paddle_timm.trace_utils import _assert
from ppsci.utils.initializer import trunc_normal_

__all__ = [
    "Attention",
    "Block",
    "DropPath",
    "drop_path",
    "Format",
    "to_1tuple",
    "to_2tuple",
    "to_ntuple",
    "Mlp",
    "PatchEmbed",
    "_assert",
    "trunc_normal_",
]

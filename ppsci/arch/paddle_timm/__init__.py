# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal set of timm components needed by the ClimaX backbone.

# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24), converted to paddle and trimmed to the code paths ClimaX
exercises (manual attention branch, no qk-norm / layer scale).
"""

from ppsci.utils.initializer import trunc_normal_

from .attention import Attention
from .block import Block
from .drop import DropPath
from .drop import drop_path
from .format import Format
from .helpers import to_1tuple
from .helpers import to_2tuple
from .helpers import to_ntuple
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .trace_utils import _assert

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

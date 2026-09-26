# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Trace assertions

# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24),
trimmed to the _assert helper used by patch_embed.
"""


def _assert(condition: bool, message: str):
    assert condition, message

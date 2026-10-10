# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2020 Ross Wightman. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
# Adapted from https://github.com/huggingface/pytorch-image-models (timm 1.0.24).
"""

from __future__ import annotations

import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_ntuple = _ntuple

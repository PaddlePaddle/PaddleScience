import itertools
import numbers
from typing import Any

import paddle
from paddle import Tensor


def repeat(src: Any, length: int) -> Any:
    if src is None:
        return None

    if isinstance(src, Tensor):
        if src.numel() == 1:
            return src.tile([length])

        if src.numel() > length:
            return src[:length]

        if src.numel() < length:
            last_elem = src[-1].unsqueeze(0)
            padding = last_elem.tile([length - src.numel()])
            return paddle.concat([src, padding])

        return src

    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))

    if len(src) > length:
        return src[:length]

    if len(src) < length:
        return src + list(itertools.repeat(src[-1], length - len(src)))

    return src

from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.typing import TensorFrame


def mask_select(src: Tensor, dim: int, mask: Tensor) -> Tensor:
    """Returns a new tensor which masks the src tensor along the
    dimension dim according to the boolean mask mask."""
    assert mask.ndim == 1

    if isinstance(src, TensorFrame):
        assert dim == 0 and src.shape[0] == mask.numel()
        return src[mask]

    assert src.shape[dim] == mask.numel()
    dim = dim + src.ndim if dim < 0 else dim
    assert dim >= 0 and dim < src.ndim

    src = paddle.transpose(src, perm=[dim] + [i for i in range(src.ndim) if i != dim]) if dim != 0 else src
    out = src[mask]
    out = paddle.transpose(out, perm=[dim] + [i for i in range(out.ndim) if i != dim]) if dim != 0 else out

    return out


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    """Converts indices to a mask representation."""
    index = index.reshape([-1])
    size = int(index.max().item()) + 1 if size is None else size
    mask = paddle.zeros([size], dtype=paddle.bool)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    """Converts a mask to an index representation."""
    return paddle.nonzero(mask).reshape([-1])
